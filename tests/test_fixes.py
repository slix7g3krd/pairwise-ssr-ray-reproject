"""SSR 六个核心缺陷修复的行为回归检查。"""

import pytest

from graphics import ssr_ray_reproject as ssr


def test_offscreen_ray_fails_instead_of_clamping_to_edge():
    assert ssr.offscreen_reject((1.2, 0.5))["rejected"] is True
    assert ssr.offscreen_reject((-0.01, 0.5))["rejected"] is True
    assert ssr.offscreen_reject((0.5, 1.01))["rejected"] is True
    assert ssr.offscreen_reject((0.5, 0.5))["accepted"] is True

    path = [(0.2, 0.2), (0.6, 0.4), (1.05, 0.8)]
    report = ssr.offscreen_reject(path)
    assert report["rejected"] is True
    assert report["index"] == 2


def test_roughness_selects_trace_layer():
    mirror = ssr.roughness_layered(0.08)
    glossy = ssr.roughness_layered(0.32)
    rough = ssr.roughness_layered(0.9)

    assert mirror["specular_trace"] is True
    assert mirror["resolution_scale"] == 1.0
    assert glossy["resolution_scale"] == 0.5
    assert glossy["blur_radius"] >= 1
    assert rough["specular_trace"] is False
    assert rough["probe_fallback"] is True
    assert rough["blur_radius"] > glossy["blur_radius"]


def test_history_requires_depth_match():
    good = ssr.history_depth_check(2.001, 2.0, relative_tolerance=0.01)
    bad = ssr.history_depth_check(2.0, 8.0)

    assert good["valid"] is True
    assert good["weight"] == pytest.approx(1.0)
    assert bad["valid"] is False
    assert bad["weight"] == 0.0
    assert bad["reason"] == "depth_mismatch"


def test_bilateral_upsample_preserves_depth_edges():
    low = [[1.0, 0.0], [1.0, 0.0]]
    full_depth = [
        [1.0, 1.0, 5.0, 5.0],
        [1.0, 1.0, 5.0, 5.0],
        [1.0, 1.0, 5.0, 5.0],
        [1.0, 1.0, 5.0, 5.0],
    ]

    out = ssr.bilateral_upsample(low, full_depth)
    assert len(out) == 4 and len(out[0]) == 4
    assert out[0][0] == pytest.approx(1.0)
    assert out[0][3] == pytest.approx(0.0)
    # 深度边缘附近不能把右侧 0 混到左侧近景像素。
    assert out[1][1] > 0.9
    assert out[1][2] < 0.1

    flat_depth = [[1.0] * 4 for _ in range(4)]
    smooth = ssr.bilateral_upsample([[0.0, 1.0], [1.0, 0.0]], flat_depth)
    assert 0.0 < smooth[1][1] < 1.0


def test_same_frame_capture_and_stale_detection():
    identity = ssr.identity_matrix()
    packet = ssr.same_frame_matrix(12, identity, identity)
    assert packet["frame"] == 12
    assert packet["camera_frame"] == packet["trace_frame"] == 12
    assert packet["lag_frames"] == 0
    assert packet["aligned"] is True
    assert len(packet["token"]) >= 8

    stale = ssr.same_frame_matrix({"frame": 11}, {"frame": 12})
    assert stale["aligned"] is False
    assert stale["ok"] is False
    assert stale["lag_frames"] == 1


def test_trace_returns_offscreen_failure_and_validates_depth_hit():
    depth = [[10.0] * 8 for _ in range(8)]
    offscreen = ssr.trace(
        origin_uv=(0.8, 0.5),
        direction_uv=(0.5, 0.0),
        origin_depth=0.0,
        depth=depth,
        max_steps=4,
        thickness=0.02,
    )
    assert offscreen["hit"] is False
    assert offscreen["reason"] == "offscreen"
    assert offscreen["step"] is not None

    good_hit = ssr.hit_validate(
        {"uv": (0.5, 0.5), "hit_depth": 2.001},
        2.0,
        thickness=0.01,
    )
    bad_hit = ssr.hit_validate(
        {"uv": (0.5, 0.5), "hit_depth": 2.0},
        9.0,
        thickness=0.01,
    )
    assert good_hit["valid"] is True
    assert bad_hit["valid"] is False
    assert "depth_inconsistent" in bad_hit["reasons"]


def test_support_contracts_are_stable():
    assert ssr.determinism(11)["values"] == ssr.determinism(11)["values"]
    assert ssr.validation(1)["ok"] is True
    assert ssr.audit(1)["event_count"] == 1
    assert ssr.fallback(1)["mode"].startswith("uniform")
    assert ssr.metrics(1.0)["rms"] == pytest.approx(1.0)
    assert ssr.thickness(2.0, 5.0) == pytest.approx(3.0)
    assert int(ssr.step_budget(1)) == 1
    assert ssr.blur(1.0) == pytest.approx(1.0)
