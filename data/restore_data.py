"""Verify and restore the split study-data archives (Python 3.9+, no packages)."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

CHUNK = 4 * 1024 * 1024


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if (p.is_absolute() or len(p.parts) < 3 or p.parts[0] != "data"
            or any(part in (".", "..", "") or ":" in part for part in p.parts)
            or "\\" in name or str(p) != name):
        raise ValueError(f"Unsafe archive path: {name}")
    return p


import bisect
from contextlib import contextmanager
import io


def find_asset(root, name):
    if Path(name).name != name or "/" in name or "\\" in name or ":" in name:
        raise ValueError(f"Invalid asset name: {name}")
    direct = root / name
    if direct.is_file():
        return direct
    matches = [p for p in root.glob("batch_*/" + name) if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate file: {name}. Download all parts into the same folder.")
    return matches[0]


class SplitReader(io.RawIOBase):
    """Expose multiple physical files as one seekable, read-only ZIP stream."""

    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.ends = []
        total = 0
        for path in paths:
            total += path.stat().st_size
            self.ends.append(total)
        self.length = total
        self.position = 0
        self.stream = None
        self.active = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.position + offset
        elif whence == 2:
            position = self.length + offset
        else:
            raise ValueError("Invalid seek origin")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read from closed archive")
        available = max(0, self.length - self.position)
        remaining = available if size is None or size < 0 else min(size, available)
        result = bytearray()
        while remaining:
            index = bisect.bisect_right(self.ends, self.position)
            start = self.ends[index - 1] if index else 0
            if self.active != index:
                if self.stream is not None:
                    self.stream.close()
                self.stream = self.paths[index].open("rb")
                self.active = index
            self.stream.seek(self.position - start)
            block = self.stream.read(min(remaining, self.ends[index] - self.position))
            if not block:
                raise OSError(f"Truncated archive part: {self.paths[index].name}")
            result.extend(block)
            self.position += len(block)
            remaining -= len(block)
        return bytes(result)

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()


@contextmanager
def open_archive(root, entry):
    paths = [find_asset(root, part["name"]) for part in entry["parts"]]
    with SplitReader(paths) as stream:
        with zipfile.ZipFile(stream) as archive:
            yield archive


def validate_manifest(manifest):
    names = []
    for entry in manifest["archives"]:
        parts = entry["parts"]
        if not parts or sum(p["bytes"] for p in parts) != entry["bytes"]:
            raise ValueError(f"Invalid part sizes: {entry['name']}")
        for index, part in enumerate(parts, 1):
            if part["name"] != f"{entry['name']}.{index:03d}" or not 0 < part["bytes"] <= 24_000_000:
                raise ValueError(f"Invalid part metadata: {part['name']}")
            names.append(part["name"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive parts")


def validate_parts(root, entry):
    combined = hashlib.sha256()
    for part in entry["parts"]:
        path = find_asset(root, part["name"])
        if path.stat().st_size != part["bytes"]:
            raise ValueError(f"Size mismatch: {part['name']}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(CHUNK), b""):
                digest.update(block)
                combined.update(block)
        if digest.hexdigest() != part["sha256"]:
            raise ValueError(f"Checksum mismatch: {part['name']}")
    if combined.hexdigest() != entry["sha256"]:
        raise ValueError(f"Combined ZIP checksum mismatch: {entry['name']}")


def validate(archive_dir, manifest):
    records = manifest["files"]
    expected = {r["path"]: r for r in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate paths in manifest")
    archive_names = [a["name"] for a in manifest["archives"]]
    if len(set(archive_names)) != len(archive_names):
        raise ValueError("Duplicate archive names")
    if any(r["archive"] not in archive_names for r in records):
        raise ValueError("Manifest refers to an unknown archive")
    for name in expected:
        safe_relative(name)
    count = 0
    total = 0
    for entry in manifest["archives"]:
        name = entry["name"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid archive filename")
        print(f"Checking {name} ({len(entry['parts'])} parts) ...", flush=True)
        validate_parts(archive_dir, entry)
        subset = {key: r for key, r in expected.items() if r["archive"] == name}
        with open_archive(archive_dir, entry) as archive:
            infos = archive.infolist()
            if len(infos) != len(subset) or {i.filename for i in infos} != set(subset):
                raise ValueError(f"Archive members differ from manifest: {name}")
            for info in infos:
                record = subset[info.filename]
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Unexpected non-file member: {info.filename}")
                h = hashlib.sha256()
                size = 0
                with archive.open(info) as stream:
                    for block in iter(lambda: stream.read(CHUNK), b""):
                        h.update(block)
                        size += len(block)
                if size != record["bytes"] or h.hexdigest() != record["sha256"]:
                    raise ValueError(f"File checksum mismatch: {info.filename}")
                count += 1
                total += size
    if count != manifest["file_count"] or total != manifest["uncompressed_bytes"]:
        raise ValueError("Manifest totals do not match")
    print(f"Verified {count} files; {total:,} uncompressed bytes.", flush=True)
    return expected


def destination(root, name):
    relative = safe_relative(name)
    target = root.joinpath(*relative.parts)
    # Reject symlinks/junctions that redirect extraction outside the destination.
    target.resolve().relative_to(root)
    return target


def restore(archive_dir, root, manifest, expected):
    root = root.resolve()
    existing = set()
    # Check all conflicts before writing any data files.
    for name, record in expected.items():
        target = destination(root, name)
        if target.exists():
            if (not target.is_file() or target.stat().st_size != record["bytes"]
                    or file_hash(target) != record["sha256"]):
                raise ValueError(f"Existing file differs; no files overwritten: {target}")
            existing.add(name)
        for parent in target.parents:
            if parent == root:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Destination parent is not a directory: {parent}")
    written = 0
    for item in manifest["archives"]:
        print(f"Restoring {item['name']} ...", flush=True)
        with open_archive(archive_dir, item) as archive:
            for info in archive.infolist():
                if info.filename in existing:
                    continue
                target = destination(root, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                record = expected[info.filename]
                h = hashlib.sha256()
                size = 0
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restore_", delete=False) as out:
                        temporary = Path(out.name)
                        with archive.open(info) as source:
                            for block in iter(lambda: source.read(CHUNK), b""):
                                out.write(block)
                                h.update(block)
                                size += len(block)
                    if size != record["bytes"] or h.hexdigest() != record["sha256"]:
                        raise ValueError(f"Extracted file failed verification: {info.filename}")
                    if target.exists():
                        raise ValueError(f"Destination appeared during extraction: {target}")
                    os.replace(temporary, target)
                    written += 1
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
    print(f"Restored {written} files; skipped {len(existing)} identical files.")
    print(f"Data directory: {root / 'data'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the split files (or batch folders) and data_manifest.json")
    parser.add_argument("--output", type=Path, default=Path.cwd(),
                        help="Destination repository root (data/ will be created inside it)")
    parser.add_argument("--verify-only", action="store_true", help="Verify all files without extracting")
    args = parser.parse_args()
    try:
        manifest = json.loads(find_asset(args.archives, "data_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 2:
            raise ValueError("Unsupported manifest format")
        validate_manifest(manifest)
        expected = validate(args.archives, manifest)
        if not args.verify_only:
            restore(args.archives, args.output, manifest, expected)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
