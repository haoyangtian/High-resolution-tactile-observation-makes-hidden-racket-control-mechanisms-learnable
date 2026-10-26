"""Software checks on synthetic inputs only; no manuscript statistics or data."""
from __future__ import annotations

import importlib
import importlib.util
import math
import unittest

import numpy as np
import torch

from tactile_components.organization.graph_blocks import GCNBlock, SAGEBlock, GATBlock
from tactile_components.organization.control_model import GraphTemporalSplitAE, BiomimeticImpedanceDecoder
from tactile_components.posture.pose_model import GNNSuite
from tactile_components.posture.hand_kinematics import PlanarHandKinematics
from tactile_components.posture import grip_geometry as geometry
from tactile_components.spatial import graph_topology as graph
from tactile_components.spatial import roi_partition as roi
from tactile_components.spatial.spatial_factorization import GraphSynergyNMF
from tactile_components.temporal.temporal_attention import PersonSpecificTemporalAttention, WindowNormalizedTCNBlock
from tactile_components.temporal.temporal_graph import build_temporal_forest
from tactile_components.skill.multimodal_model import ActionGNNSuite
from tactile_components.learning.reward_model import RewardModel, ImuPriorModel


class Components(unittest.TestCase):
    def setUp(self):
        # Software-test seeds and shapes do not specify study configurations.
        torch.manual_seed(17)
        self.coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32)
        self.adj = graph.build_knn_adj(self.coords, k=2)

    def test_all_component_modules_import(self):
        modules = {
            "posture": ["pose_model", "hand_kinematics", "grip_geometry"],
            "organization": ["control_model", "graph_blocks"],
            "spatial": ["graph_topology", "roi_partition", "spatial_factorization"],
            "temporal": ["temporal_attention", "temporal_graph"],
            "skill": ["multimodal_model"],
            "learning": ["reward_model"],
        }
        for group, names in modules.items():
            for name in names:
                with self.subTest(module=name):
                    importlib.import_module(f"tactile_components.{group}.{name}")

    def test_graph_construction_singleton_empty_and_zero_neighbors(self):
        for fn in [graph.build_knn_adj, graph.build_radius_adj, graph.build_mst_adj,
                   graph.build_mst_knn_hybrid, graph.build_fully_connected]:
            with self.subTest(builder=fn.__name__):
                for loop in (True, False):
                    a = fn(np.zeros((1, 2)), self_loop=loop)
                    self.assertEqual(a.shape, (1, 1))
                    self.assertEqual(a.item(), float(loop))
                    self.assertEqual(fn(np.empty((0, 2)), self_loop=loop).shape, (0, 0))
        torch.testing.assert_close(graph.build_knn_adj(self.coords, k=0), torch.eye(5))
        with self.assertRaises(ValueError):
            graph.build_knn_adj(self.coords, k=-1)

    def test_graph_edges_and_mst_connectivity(self):
        import networkx as nx
        for fn in [graph.build_knn_adj, graph.build_radius_adj, graph.build_mst_adj,
                   graph.build_mst_knn_hybrid, graph.build_fully_connected]:
            a = fn(self.coords, self_loop=False)
            torch.testing.assert_close(a, a.T)
            torch.testing.assert_close(a.diag(), torch.zeros(5))
            self.assertTrue(bool(((a == 0) | (a == 1)).all()))
        mst = nx.from_numpy_array(graph.build_mst_adj(self.coords, self_loop=False).numpy())
        self.assertTrue(nx.is_tree(mst))

    def test_graph_blocks_forward_backward_and_non_contiguous(self):
        blocks = [GCNBlock(7, 16, dropout=0., style="stacked"),
                  GCNBlock(7, 16, dropout=0., style="residual"),
                  SAGEBlock(7, 16, dropout=0., bias=False),
                  GATBlock(7, 16, heads=2, dropout=0.)]
        for block in blocks:
            with self.subTest(block=type(block).__name__):
                block.eval()
                x = torch.randn(2, 3, 7, 5).transpose(-1, -2).requires_grad_()
                y = block(x, self.adj)
                self.assertEqual(y.shape, (2, 3, 5, 16))
                torch.testing.assert_close(y, block(x.contiguous(), self.adj))
                y.square().sum().backward()
                self.assertTrue(bool(torch.isfinite(y).all()))
                self.assertTrue(bool(torch.isfinite(x.grad).all()))
                with self.assertRaises(ValueError): block(x, torch.ones(5, 4))
                with self.assertRaises(ValueError): block(x, torch.full((5, 5), float("nan")))

    def test_attention_rejects_empty_rows_and_invalid_heads(self):
        with self.assertRaises(ValueError): GATBlock(7, 16, heads=0)
        block = GATBlock(7, 16, heads=2)
        with self.assertRaises(ValueError): block(torch.randn(2, 5, 7), torch.zeros(5, 5))

    def test_pose_model_all_spatial_and_temporal_branches(self):
        for spatial in ("gcn", "sage", "gat"):
            for temporal in ("mean", "last", "attn", "lstm"):
                with self.subTest(spatial=spatial, temporal=temporal):
                    model = GNNSuite(5, hidden=16, spatial=spatial, temporal=temporal, dropout=0.).eval()
                    y = model(torch.randn(2, 7, 5, 1), self.adj)
                    self.assertEqual(y.shape, (2, 2))
                    self.assertTrue(bool(torch.isfinite(y).all()))

    def test_pose_model_coordinate_broadcasting(self):
        model = GNNSuite(5, in_ch=4, hidden=16, dropout=0.).eval()
        x = torch.randn(2, 7, 5, 1)
        xyz = torch.randn(5, 3)
        expected = model(x, self.adj, xyz)
        for pos in (xyz.expand(7, 5, 3), xyz.expand(2, 5, 3), xyz.expand(2, 7, 5, 3)):
            torch.testing.assert_close(model(x, self.adj, pos), expected)

    def test_planar_kinematics_known_geometry(self):
        h = PlanarHandKinematics
        np.testing.assert_allclose(h._forward_thumb_2seg(2., 3., 0., 0.), [2, 0, 5, 0])
        np.testing.assert_allclose(h._forward_finger_3seg(2., 3., 4., 0., 0., 0., "mid"), [3.5, 0])
        np.testing.assert_allclose(h._forward_finger_4seg(1., 2., 3., 4., 0., 0., 0.), [1, 0, 3, 0, 6, 0, 10, 0])
        rotated = h._rotate_vec_about_axis(np.array([1., 0., 0.]), np.array([0., 0., 1.]), math.pi / 2)
        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-6)
        origin = np.array([2., 2., 2.])
        np.testing.assert_allclose(h._rotate_point_about_axis(origin + [1, 0, 0], origin, np.array([0., 0., 1.]), math.pi / 2), origin + [0, 1, 0], atol=1e-6)

    def test_similarity_transform_known_rotation_scale_translation(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        rot = geometry._rotation_matrix_from_axis_angle(np.array([0., 0., 1.]), .7)
        target = 1.7 * (source @ rot.T) + np.array([2., -3., 4.])
        scale, rotation, translation = geometry._fit_similarity_transform(source, target)
        np.testing.assert_allclose(scale * (source @ rotation.T) + translation, target, atol=2e-6)

    def test_explicit_partition_backend_and_invalid_inputs(self):
        z = np.eye(4)
        edges, weights = [(0, 1), (2, 3)], [1., 1.]
        labels, score = roi._partition_graph(z, edges, weights, 1., 17, backend="louvain")
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertTrue(np.isfinite(score))
        with self.assertRaises(ValueError): roi._partition_graph(z, edges, weights, 1., 17, backend="automatic")
        with self.assertRaises(ValueError): roi._partition_graph(z, edges, [-1., 1.], 1., 17)
        with self.assertRaises(ValueError): roi._partition_graph(z, [(0, 9)], [1.], 1., 17)
        empty, _ = roi._partition_graph(z, [], [], 1., 17)
        np.testing.assert_array_equal(empty, np.arange(4))

    def test_leiden_or_explicit_missing_dependency(self):
        if importlib.util.find_spec("igraph") and importlib.util.find_spec("leidenalg"):
            labels, score = roi._partition_graph(np.eye(4), [(0, 1), (2, 3)], [1., 1.], 1., 17)
            self.assertEqual(labels[0], labels[1])
            self.assertNotEqual(labels[0], labels[2])
            self.assertTrue(np.isfinite(score))
        else:
            with self.assertRaises(ImportError):
                roi._partition_graph(np.eye(4), [(0, 1), (2, 3)], [1., 1.], 1., 17)

    def test_spatial_factorization_is_nonnegative_and_differentiable(self):
        model = GraphSynergyNMF(9, 5, 3)
        result = model()
        self.assertIsInstance(result, tuple)
        for item in result:
            self.assertTrue(bool((item >= 0).all()))
            self.assertTrue(bool(torch.isfinite(item).all()))
        sum(item.sum() for item in result).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_temporal_weights_normalized_and_independent_of_level_metadata(self):
        model = PersonSpecificTemporalAttention(1, 3, 2, n_levels=4)
        with torch.no_grad(): model.person_delta.weight.copy_(torch.tensor([[1., 2., 3.], [3., 2., 1.]]))
        a = model(torch.tensor([0, 1]), level_id=torch.tensor([0, 3]))
        b = model(torch.tensor([0, 1]), level_id=torch.tensor([2, 1]))
        torch.testing.assert_close(a['weights'], b['weights'], rtol=0, atol=0)
        torch.testing.assert_close(a['weights'].sum(-1), torch.ones(2))
        torch.testing.assert_close(a['drive_weights'], a['grip_weights'])
        model.set_force_uniform_attention(True)
        torch.testing.assert_close(model(torch.tensor([0, 1]))['weights'], torch.full((2, 3), 1 / 3))

    def test_window_tcn_with_separate_past_only_windows(self):
        model = WindowNormalizedTCNBlock(5, 3, 2, dropout_p=0.).eval()
        x = torch.randn(2, 5, 15)
        changed = x.clone(); changed[:, :, 9:] += 100
        def predict_by_window(values):
            windows = torch.nn.functional.pad(values, (6, 0)).unfold(2, 7, 1)
            windows = windows.permute(0, 2, 1, 3).reshape(30, 5, 7)
            return model(windows)[:, :, -1].reshape(2, 15, 5)
        torch.testing.assert_close(predict_by_window(x)[:, :9], predict_by_window(changed)[:, :9], rtol=0, atol=0)

    def make_control_model(self):
        return GraphTemporalSplitAE(5, np.eye(5, dtype=np.float32), 16, 4, 4,
                                    n_persons=2, delay_kmin=1, delay_kmax=3,
                                    local_window=1, input_dropout_p=0., tcn_dropout_p=0.).eval()

    def test_control_model_complete_component_path_and_backward(self):
        model = self.make_control_model()
        x = torch.randn(2, 9, 5, requires_grad=True)
        state = model.encode(x, torch.randn(2, 9, 6))
        self.assertTrue(bool((state['z_drive_seq'] > 0).all()))
        self.assertEqual(model.decode(state['z_drive_seq'], state['z_grip_seq']).shape, (2, 9, 5))
        out = model.predict_next_pose_velocity(torch.randn(2, 9, 3), torch.randn(2, 9, 3),
            state['z_drive_seq'], state['z_grip_seq'], imu_seq=torch.randn(2, 9, 6), person_id=torch.tensor([0, 1]),
            torque_target_center=torch.zeros(3), torque_target_half_range=torch.ones(3))
        self.assertEqual(out['pose_next_pred'].shape, (2, 8, 3))
        self.assertTrue(bool(((out['k_grip'] >= 0) & (out['k_grip'] <= 1)).all()))
        torch.testing.assert_close(out['delta_v_pred'], (1 - out['k_grip']) * out['delta_v_drive'])
        out['pose_next_pred'].square().sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_current_encode_is_framewise_and_does_not_call_legacy_context(self):
        model = self.make_control_model()
        def unexpected(*args, **kwargs):
            raise AssertionError("Current encoder called a legacy context branch")
        handles = [part.register_forward_pre_hook(unexpected) for part in
                   [*model.tcn_blocks, model.context_conv1, model.context_conv2,
                    model.imu_step_proj, model.imu_context_proj, model.film_mlp]]
        try:
            x = torch.randn(2, 9, 5)
            changed = x.clone(); changed[:, 6:] += 100
            a, b = model.encode(x), model.encode(changed)
            for key in ['z_drive_seq', 'z_grip_seq', 's_perc_seq']:
                torch.testing.assert_close(a[key][:, :6], b[key][:, :6], rtol=0, atol=0)
        finally:
            for handle in handles: handle.remove()

    def test_delayed_control_buffer_uses_only_specified_past(self):
        model = self.make_control_model()
        z = torch.arange(9, dtype=torch.float32).view(1, 9, 1).expand(2, 9, 4)
        out = model.aggregate_delayed_intents(z, z, person_id=torch.tensor([0, 1]))
        torch.testing.assert_close(out['z_drive_effective_seq'][:, 5, :], torch.full((2, 4), 3.))
        self.assertFalse(bool(out['intent_window_valid'][:, :3].any()))
        self.assertTrue(bool(out['intent_window_valid'][:, 3:].all()))

    def test_temporal_forest_is_acyclic_and_preserves_supplied_lags(self):
        import networkx as nx
        w = np.array([[0, 3, 1], [0, 0, 2], [0, 0, 0]], dtype=np.float32)
        lags = np.array([[0, 2, 3], [0, 0, 1], [0, 0, 0]])
        rows, rank, metadata = build_temporal_forest(w, lags, ['a', 'b', 'c'], [0, 1, 2], None, 7.)
        g = nx.DiGraph([(r['src_idx'], r['dst_idx']) for r in rows])
        self.assertTrue(nx.is_directed_acyclic_graph(g))
        self.assertEqual(metadata['edge_count'], len(rows))
        for row in rows:
            self.assertEqual(row['lag_boot_median_ms'], float(lags[row['src_idx'], row['dst_idx']]) * 7.)
        with self.assertRaises(ValueError):
            build_temporal_forest(w, lags, ['a', 'b', 'c'], [0, 1, 2], None, 7., nearest_mode='invalid')

    def test_multimodal_model_branches_and_optional_features(self):
        for spatial in ('gcn', 'gnn', 'attention-gnn'):
            for temporal in ('mean', 'last', 'attn', 'lstm'):
                model = ActionGNNSuite(5, hidden=16, spatial=spatial, temporal=temporal,
                    num_classes=3, dropout=0., use_node_attn=True).eval()
                logits, features, attention = model(torch.randn(2, 7, 5), torch.randn(2, 7, 6), self.adj,
                    return_feat=True, return_node_attn=True)
                self.assertEqual(logits.shape, (2, 3))
                self.assertTrue(bool(torch.isfinite(features).all()))
                torch.testing.assert_close(attention.sum(-1), torch.ones(2, 7))

    def test_reward_and_motion_prior(self):
        tac, imu = torch.randn(2, 7, 5), torch.randn(2, 7, 6)
        phase, skill = torch.linspace(0, 1, 7).view(7, 1), torch.tensor([0, 1])
        reward = RewardModel(5, 6, 2, hidden=16, skill_emb=4)
        sequence, total = reward(tac, imu, skill, phase)
        torch.testing.assert_close(total, sequence.sum(-1))
        prior = ImuPriorModel(5, 2, hidden=16, skill_emb=4)
        mean, logstd = prior(tac, skill, phase)
        self.assertEqual(mean.shape, imu.shape)
        self.assertEqual(logstd.shape, imu.shape)
        self.assertTrue(bool(torch.isfinite(logstd).all()))


if __name__ == '__main__':
    unittest.main()
