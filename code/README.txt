Tactile model and algorithm components

Release scope

This repository contains selected usable model and algorithm components,
including numeric constants and constructor defaults. It is a partial source
release, not an end-to-end reproduction package for the manuscript.

Statistical inference, participant/group-level manuscript metric aggregation,
figure-generation scripts, participant identities, trained weights and
complete experiment orchestration are not included in this code directory.
The separately released study recordings and restoration instructions are in
../data/. The synthetic
examples and tests verify software behavior only; they do not reproduce or
validate the reported scientific results.

Constructor defaults are library defaults from the source implementation.
They are not asserted to be the frozen settings of every manuscript experiment.
Any additional caller-supplied configuration must be documented when using the
components in an experiment. No hidden parameter recovery table is required.

Install and verify

From this directory, in a Python environment:

    python -m pip install .
    python -m unittest discover -s tests -v
    python examples/synthetic_demo.py

To use the Leiden partition backend:

    python -m pip install ".[leiden]"

Python 3.10 or later is declared by the package. The actual verification
environment and test results are recorded in VALIDATION.json; compatibility
with other environments is not established by that check. PyTorch installation
may differ by hardware. All included tests use CPU and synthetic data.

Directory and import mapping

The six existing directory labels are preserved. Setuptools maps them to
standard import names when the package is installed:

    Posture reconstruction      tactile_components.posture
    Organization                tactile_components.organization
    Spatial realization         tactile_components.spatial
    Temporal realization        tactile_components.temporal
    Skill level estimation      tactile_components.skill
    Interactive skill leraning  tactile_components.learning

Example imports:

    from tactile_components.posture.pose_model import GNNSuite
    from tactile_components.organization.control_model import GraphTemporalSplitAE
    from tactile_components.spatial.graph_topology import build_knn_adj
    from tactile_components.temporal.temporal_attention import PersonSpecificTemporalAttention
    from tactile_components.skill.multimodal_model import ActionGNNSuite
    from tactile_components.learning.reward_model import RewardModel

Relationship to the manuscript

    Posture reconstruction: main Fig. 2 and Supplementary Methods S7.
    Organization: drive/grip control (main Figs. 2-3).
    Spatial realization: force factorization and contact regions (Figs. 3-4).
    Temporal realization: delayed control and regional ordering (Figs. 3-4).
    Skill level estimation: proficiency-level feedback (main Fig. 5).
    Interactive skill leraning: reward and motion-prior architectures
        for motion guidance (Supplementary Methods S6).

These mappings identify component roles, not complete figure-reproduction
commands. Read the category README files for module coverage. PlanarHandKinematics
contains selected static geometric helpers, not the complete hand pose solver.
The temporal forest component accepts supplied edge weights and delays; it does
not estimate transfer-entropy significance or recreate the final cohort-level
ordering analysis. The graph partition backend must be selected explicitly;
unavailable Leiden dependencies do not trigger an automatic alternative.

WindowNormalizedTCNBlock retains a historical window-wide normalization layer.
It is not a strictly causal block when given an arbitrary complete sequence;
causal use requires a separate past-only input window per prediction. The
current GraphTemporalSplitAE.encode method uses its frame-wise encoder and does
not call the legacy context/TCN branches. Tests cover this distinction. This is
not an evaluation of every historical model configuration or checkpoint.

SOURCE_MANIFEST.json records extraction origins and intentional adaptations.
CHANGELOG.txt describes differences from the previous redacted excerpt package.
The source study files were not edited when this package was prepared.

中文说明

本目录是可使用的模型与算法组件包，数值参数和默认值已恢复。论文的统计
检验、参与者/组别层面的指标汇总、绘图、训练权重和完整实验调度不随附。
真实传感器记录及恢复说明位于 ../data/。本组件包不能单独复现论文主要结果。

示例全部使用合成输入。测试通过只表示已检查的组件功能符合预期，不表示
论文结论已被复现，或所有参数组合都已验证。源码默认值不等于所有论文
实验的最终冻结配置。公开范围说明参见 CODE_AVAILABILITY.txt。
