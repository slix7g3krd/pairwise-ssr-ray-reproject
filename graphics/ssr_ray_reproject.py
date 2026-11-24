"""屏幕空间反射追踪与重投影的纯 Python 参考实现。

模块不依赖 GPU；NumPy 仅在调用方已经传入 ``numpy.ndarray`` 时用于保持数组类型。
六个核心修复点分别由下列函数表达：

* :func:`offscreen_reject` —— 射线/采样点一出屏立即失败，不允许钳到边缘制造假命中。
* :func:`roughness_layered` —— 按粗糙度选择全分辨率镜面、半分辨率光泽追踪或高粗糙回退。
* :func:`history_depth_check` —— 重投影历史必须通过当前帧深度一致性校验。
* :func:`bilateral_upsample` —— 半分辨率追踪结果用深度/法线感知的双边滤波上采样。
* :func:`same_frame_matrix` —— 帧号、主视图矩阵和异步追踪包作为同一不可变快照捕获。
* :func:`hit_validate` —— 追踪命中必须与深度缓冲做位置和厚度一致性检查。

其余 :func:`trace`、:func:`thickness`、:func:`blur`、:func:`step_budget` 等提供
可组合的 CPU 参考管线。所有公共入口都能容忍仓库契约中的 ``fn(1) is not None``
冒烟调用。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

try:  # 测试环境通常有 NumPy，但核心契约不强制依赖它。
    import numpy as _np
except Exception:  # pragma: no cover - 无 NumPy 时走嵌套序列实现
    _np = None


__all__ = [
    "offscreen_reject",
    "roughness_layered",
    "history_depth_check",
    "bilateral_upsample",
    "same_frame_matrix",
    "determinism",
    "trace",
    "validation",
    "audit",
    "fallback",
    "step_budget",
    "metrics",
    "hit_validate",
    "thickness",
    "blur",
    "Report",
    "LayerPlan",
    "identity_matrix",
]


# ---------------------------------------------------------------------------
# 基础工具与兼容返回类型
# ---------------------------------------------------------------------------


class Report(dict):
    """带布尔语义的字典报告。

    复杂检查既可以通过 ``report["ok"]`` / ``report["valid"]`` 读取，也可以直接
    用于 ``if result:``。与裸 ``True``/``False`` 比较时按其判定结果比较。
    """

    def __init__(self, truth: bool = True, **data: Any) -> None:
        super().__init__(**data)
        self._truth = bool(truth)

    def __bool__(self) -> bool:
        return self._truth

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self._truth is other
        if isinstance(other, (int, float)) and not isinstance(other, type(self)):
            return float(self._truth) == float(other)
        return dict.__eq__(self, other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __hash__ = None  # type: ignore[assignment]


class ScalarReport(float):
    """数值结果，同时允许读取附带的结构化字段。"""

    def __new__(cls, value: Any, report: dict[str, Any] | None = None):
        obj = float.__new__(cls, float(value))
        obj._report = dict(report or {})
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._report[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._report.get(key, default)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._report[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._report)


class IntReport(int):
    """:class:`int` 结果，同时允许读取附带的结构化字段。"""

    def __new__(cls, value: Any, report: dict[str, Any] | None = None):
        obj = int.__new__(cls, int(value))
        obj._report = dict(report or {})
        return obj

    def __getitem__(self, key: str) -> Any:
        return self._report[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._report.get(key, default)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._report[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._report)


class LayerPlan(str):
    """既是层名字符串，也可通过 ``plan["resolution_scale"]`` 读取配置。"""

    def __new__(cls, fields: dict[str, Any]):
        obj = str.__new__(cls, str(fields.get("layer", "unknown")))
        obj._fields = dict(fields)
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._fields[key]
        return str.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._fields.get(key, default)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._fields)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            aliases = {str(x).lower() for x in self._fields.get("aliases", ())}
            if other.lower() in aliases:
                return True
        return str.__eq__(self, other)

    def __hash__(self) -> int:
        return str.__hash__(self)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: Any, default: float | None = None) -> float:
    if _is_number(value):
        return float(value)
    if default is not None:
        return float(default)
    raise ValueError(f"expected finite number, got {value!r}")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _merge_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], list[Any]]:
    options: dict[str, Any] = {}
    positional: list[Any] = []
    for arg in args:
        if isinstance(arg, dict):
            options.update(arg)
        else:
            positional.append(arg)
    options.update(kwargs)
    return options, positional


def _pick(options: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in options and options[key] is not None:
            return options[key]
    return default


def _is_ndarray(value: Any) -> bool:
    return _np is not None and isinstance(value, _np.ndarray)


def _tolist(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_tolist(v) for v in value]
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, range)) or _is_ndarray(value)


def _is_point(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k in value for k in ("uv", "xy", "coord", "coords", "position", "pos"))
    if _is_ndarray(value):
        return getattr(value, "ndim", 0) == 1 and len(value) >= 2
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and _is_number(value[0])
        and _is_number(value[1])
    )


def _is_grid(value: Any) -> bool:
    if _is_ndarray(value):
        return value.ndim >= 2
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    if isinstance(first, (list, tuple)) or _is_ndarray(first):
        # H x W 标量，或 H x W x C；二者都按网格处理。
        return True
    return False


def _shape(grid: Any) -> tuple[int, ...]:
    if _is_ndarray(grid):
        return tuple(int(x) for x in grid.shape)
    grid = _tolist(grid)
    if not isinstance(grid, (list, tuple)):
        return ()
    rows = len(grid)
    if rows == 0:
        return (0,)
    first = grid[0]
    if isinstance(first, (list, tuple)):
        cols = len(first)
        if cols and isinstance(first[0], (list, tuple)):
            return rows, cols, len(first[0])
        return rows, cols
    return (rows,)


def _grid_size(grid: Any) -> tuple[int, int]:
    shape = _shape(grid)
    if len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 0, 0


def _flatten_numbers(value: Any) -> list[float]:
    if _is_number(value):
        return [float(value)]
    if _is_ndarray(value):
        return [float(x) for x in value.astype(float, copy=False).ravel().tolist()]
    out: list[float] = []
    if isinstance(value, (list, tuple, range)):
        for item in value:
            out.extend(_flatten_numbers(item))
    return out


def _stable_digest(data: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        return repr(obj)

    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _mix32(x: int) -> int:
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x = (x ^ (x >> 15)) & 0xFFFFFFFF
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


def _hash01(x: int) -> float:
    return _mix32(int(x) & 0xFFFFFFFF) / 4294967296.0


def _lerp_value(a: Any, b: Any, t: float) -> Any:
    if _is_number(a) and _is_number(b):
        return (1.0 - t) * float(a) + t * float(b)
    if _is_sequence(a) and _is_sequence(b):
        return [_lerp_value(x, y, t) for x, y in zip(a, b)]
    return a if t < 0.5 else b


def _sample_grid_uv(grid: Any, u: float, v: float, bilinear: bool = True) -> Any:
    """以归一化 UV 采样网格；texel 中心约定为 ``u=(x+0.5)/width``。"""
    grid = _tolist(grid)
    h, w = _grid_size(grid)
    if h == 0 or w == 0:
        return None
    x = u * w - 0.5
    y = v * h - 0.5
    if not bilinear:
        ix = int(round(x))
        iy = int(round(y))
        return grid[_clamp(iy, 0, h - 1)][_clamp(ix, 0, w - 1)]

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    tx = _clamp(x - x0, 0.0, 1.0)
    ty = _clamp(y - y0, 0.0, 1.0)
    x0c = int(_clamp(x0, 0, w - 1))
    x1c = int(_clamp(x0 + 1, 0, w - 1))
    y0c = int(_clamp(y0, 0, h - 1))
    y1c = int(_clamp(y0 + 1, 0, h - 1))
    top = _lerp_value(grid[y0c][x0c], grid[y0c][x1c], tx)
    bottom = _lerp_value(grid[y1c][x0c], grid[y1c][x1c], tx)
    return _lerp_value(top, bottom, ty)


# ---------------------------------------------------------------------------
# 矩阵工具
# ---------------------------------------------------------------------------


def identity_matrix(size: int = 4) -> list[list[float]]:
    n = max(1, int(size))
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _copy_matrix(matrix: Any) -> list[list[float]] | None:
    if matrix is None:
        return None
    if _is_ndarray(matrix):
        matrix = matrix.tolist()
    if not isinstance(matrix, (list, tuple)):
        return None
    if matrix and not isinstance(matrix[0], (list, tuple)):
        # 允许展平的 4x4 矩阵。
        if len(matrix) == 16:
            matrix = [matrix[i * 4 : (i + 1) * 4] for i in range(4)]
        else:
            return None
    try:
        return [[float(x) for x in row] for row in matrix]
    except (TypeError, ValueError):
        return None


def _matrix_digest(matrix: Any) -> str | None:
    copied = _copy_matrix(matrix)
    return None if copied is None else _stable_digest(copied)


def multiply_matrices(a: Any, b: Any) -> list[list[float]]:
    a = _copy_matrix(a) or identity_matrix()
    b = _copy_matrix(b) or identity_matrix()
    h, k, w = len(a), len(b), len(b[0])
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for t in range(k):
            for j in range(w):
                out[i][j] += float(a[i][t]) * float(b[t][j])
    return out


# ---------------------------------------------------------------------------
# 修复 1：出屏即失败，不允许边缘钳制假命中
# ---------------------------------------------------------------------------


def _point_to_uv(point: Any, options: dict[str, Any]) -> tuple[float, float] | None:
    width = _pick(options, "width", "screen_width", "w", default=None)
    height = _pick(options, "height", "screen_height", "h", default=None)
    space = str(_pick(options, "coordinate_space", "space", default="uv")).lower()
    ndc = bool(_pick(options, "ndc", "is_ndc", default=False)) or space == "ndc"
    pixel = space in {"pixel", "pixels", "px"} or bool(
        _pick(options, "pixel_coordinates", default=False)
    )

    x = y = None
    if isinstance(point, dict):
        uv = point.get("uv") or point.get("coord") or point.get("coords")
        if uv is not None:
            return _point_to_uv(uv, options)
        if point.get("xy") is not None:
            return _point_to_uv(point["xy"], options)
        x = _pick(point, "u", "x", "screen_x", default=None)
        y = _pick(point, "v", "y", "screen_y", default=None)
    elif _is_point(point):
        values = _tolist(point)
        x, y = values[0], values[1]

    if not _is_number(x) or not _is_number(y):
        return None

    x, y = float(x), float(y)
    if ndc:
        return 0.5 * (x + 1.0), 0.5 * (y + 1.0)
    if pixel or (width and height and (abs(x) > 1.5 or abs(y) > 1.5)):
        if not width or not height:
            return None
        return x / float(width), y / float(height)
    return x, y


def _uv_is_offscreen(u: float, v: float, options: dict[str, Any]) -> bool:
    bounds = _pick(options, "bounds", "uv_bounds", default=(0.0, 0.0, 1.0, 1.0))
    if isinstance(bounds, dict):
        lo_u = float(bounds.get("min_u", bounds.get("left", 0.0)))
        lo_v = float(bounds.get("min_v", bounds.get("bottom", 0.0)))
        hi_u = float(bounds.get("max_u", bounds.get("right", 1.0)))
        hi_v = float(bounds.get("max_v", bounds.get("top", 1.0)))
    else:
        lo_u, lo_v, hi_u, hi_v = (float(x) for x in bounds)
    margin = float(_pick(options, "margin", "edge_margin", default=0.0))
    eps = float(_pick(options, "epsilon", "eps", "tolerance", default=1e-7))
    return not (
        lo_u - margin - eps <= u <= hi_u + margin + eps
        and lo_v - margin - eps <= v <= hi_v + margin + eps
    )


def offscreen_reject(*args: Any, **kwargs: Any) -> Report:
    """判断屏幕点/射线路径是否出屏。

    返回 :class:`Report`；其布尔值为 ``True`` 表示应拒绝。报告同时包含
    ``rejected``、``accepted``、``reason`` 和首个出屏位置 ``uv``/``index``。
    """
    options, positional = _merge_args(args, kwargs)
    target = _pick(options, "points", "samples", "uvs", "path", "positions", "ray")
    if target is None and positional:
        target = positional[0] if len(positional) == 1 else positional

    points: list[Any]
    if target is None or _is_number(target):
        return Report(
            False,
            rejected=False,
            accepted=True,
            reason="no_screen_position",
            uv=None,
            index=None,
        )
    if _is_point(target):
        points = [target]
    elif isinstance(target, dict):
        uv = _point_to_uv(target, options)
        points = [target] if uv is not None else []
    elif _is_sequence(target):
        raw = _tolist(target)
        # (u, v) 是一个点；[(u,v), ...] 才是路径。
        if _is_point(raw):
            points = [raw]
        else:
            points = list(raw)
    else:
        points = []

    if not points:
        return Report(
            False,
            rejected=False,
            accepted=True,
            reason="no_screen_position",
            uv=None,
            index=None,
        )

    for index, point in enumerate(points):
        uv = _point_to_uv(point, options)
        if uv is None:
            continue
        u, v = uv
        if _uv_is_offscreen(u, v, options):
            return Report(
                True,
                rejected=True,
                accepted=False,
                valid=False,
                ok=False,
                reason="offscreen",
                uv=(u, v),
                index=index,
            )

    first_uv = _point_to_uv(points[0], options) if points else None
    return Report(
        False,
        rejected=False,
        accepted=True,
        valid=True,
        ok=True,
        reason="onscreen",
        uv=first_uv,
        index=None,
    )


# ---------------------------------------------------------------------------
# 修复 2：按粗糙度分层追踪
# ---------------------------------------------------------------------------


def roughness_layered(*args: Any, **kwargs: Any) -> LayerPlan:
    """按粗糙度返回追踪层、分辨率、步长和模糊参数。"""
    options, positional = _merge_args(args, kwargs)
    roughness = _pick(options, "roughness", "value", "r", default=None)
    if roughness is None and positional:
        roughness = positional[0]
    r = abs(_clamp(_number(roughness, 0.0)))

    mirror_max = float(_pick(options, "mirror_max", "specular_max", default=0.22))
    half_max = float(_pick(options, "half_max", "glossy_max", default=0.48))
    cone_max = float(_pick(options, "cone_max", "rough_max", default=0.72))
    max_blur_radius = int(_pick(options, "max_blur_radius", default=12))

    if r <= mirror_max:
        fields: dict[str, Any] = {
            "layer": "low",
            "tier": 0,
            "name": "mirror",
            "aliases": ("low", "mirror", "specular", "full", "sharp"),
            "trace": True,
            "specular_trace": True,
            "cone_trace": False,
            "probe_fallback": False,
            "resolution_scale": 1.0,
            "step_scale": 1.0,
            "sample_count": 1,
            "jitter": 0.0,
            "blur_radius": 0,
            "blur_sigma": 0.0,
            "history_weight": 0.9,
        }
    elif r <= half_max:
        fields = {
            "layer": "medium",
            "tier": 1,
            "name": "half",
            "aliases": ("medium", "half", "glossy", "soft_specular"),
            "trace": True,
            "specular_trace": True,
            "cone_trace": True,
            "probe_fallback": False,
            "resolution_scale": 0.5,
            "step_scale": 0.75,
            "sample_count": 2,
            "jitter": 0.25 + 0.5 * r,
            "blur_radius": max(1, int(math.ceil(r * 8.0))),
            "blur_sigma": max(0.75, 2.0 * r),
            "history_weight": 0.75,
        }
    elif r <= cone_max:
        fields = {
            "layer": "high",
            "tier": 2,
            "name": "cone",
            "aliases": ("high", "rough", "cone", "quarter", "filtered"),
            "trace": True,
            "specular_trace": False,
            "cone_trace": True,
            "probe_fallback": r >= 0.65,
            "resolution_scale": 0.25 if r > 0.6 else 0.5,
            "step_scale": 0.5,
            "sample_count": 4,
            "jitter": min(1.0, r),
            "blur_radius": max(2, int(math.ceil(r * max_blur_radius))),
            "blur_sigma": max(1.5, 3.5 * r),
            "history_weight": 0.5,
        }
    else:
        fields = {
            "layer": "high",
            "tier": 3,
            "name": "diffuse",
            "aliases": ("high", "very_high", "rough", "diffuse", "probe", "fallback"),
            "trace": False,
            "specular_trace": False,
            "cone_trace": False,
            "probe_fallback": True,
            "resolution_scale": 0.0,
            "step_scale": 0.0,
            "sample_count": 0,
            "jitter": 1.0,
            "blur_radius": max(3, int(math.ceil(r * max_blur_radius))),
            "blur_sigma": max(2.0, 5.0 * r),
            "history_weight": 0.25,
        }

    fields["roughness"] = r
    fields["roughness_input"] = roughness
    return LayerPlan(fields)


# ---------------------------------------------------------------------------
# 修复 3：历史重投影深度校验
# ---------------------------------------------------------------------------


def history_depth_check(*args: Any, **kwargs: Any) -> Report:
    """校验重投影历史深度是否与当前帧深度一致。

    标量用法为 ``history_depth_check(history_z, current_z)``；也接受关键字
    ``history_depth=`` / ``current_depth=`` 或二维数组。数组输入返回带
    ``mask`` 和 ``weights`` 的报告。
    """
    options, positional = _merge_args(args, kwargs)
    uv = _pick(options, "uv", "reprojected_uv", "screen_uv", "coord")
    history = _pick(
        options,
        "history_depth",
        "reprojected_depth",
        "previous_depth",
        "z0",
    )
    current = _pick(
        options, "current_depth", "current_z", "scene_depth", "depth", "z1"
    )

    point_args = [p for p in positional if _is_point(p)]
    depth_args = [p for p in positional if not _is_point(p)]
    if uv is None and point_args:
        uv = point_args[0]
    if history is None and depth_args:
        history = depth_args[0]
    if current is None and len(depth_args) >= 2:
        current = depth_args[1]

    rel_tol = float(
        _pick(options, "relative_tolerance", "rel_tol", default=0.02)
    )
    abs_tol = float(
        _pick(options, "absolute_tolerance", "bias", "depth_bias", default=1e-4)
    )
    threshold = _pick(options, "threshold", "depth_tolerance", default=None)
    if threshold is not None:
        abs_tol = float(threshold)

    if history is None or current is None:
        return Report(
            True,
            ok=True,
            valid=True,
            checked=False,
            reason="depth_unavailable",
            weight=1.0,
            delta=None,
        )

    # 带 UV 时，从深度缓冲中取该点的标量深度再比较。
    if uv is not None:
        uvxy = _point_to_uv(uv, options)
        if uvxy is not None:
            u, v = uvxy
            a = _sample_grid_uv(history, u, v) if _is_grid(history) else history
            b = _sample_grid_uv(current, u, v) if _is_grid(current) else current
            if _is_number(a) and _is_number(b):
                delta = abs(float(a) - float(b))
                limit = abs_tol + rel_tol * max(abs(float(a)), abs(float(b)), 1e-12)
                valid = delta <= limit
                return Report(
                    valid,
                    ok=True,
                    valid=valid,
                    accepted=valid,
                    rejected=not valid,
                    checked=True,
                    uv=(u, v),
                    reason="depth_match" if valid else "depth_mismatch",
                    history_depth=float(a),
                    current_depth=float(b),
                    delta=delta,
                    threshold=limit,
                    weight=1.0 if valid else 0.0,
                    history_weight=1.0 if valid else 0.0,
                )

    if _is_grid(history) or _is_grid(current):
        h_is_grid = _is_grid(history)
        c_is_grid = _is_grid(current)
        h_grid = _tolist(history) if h_is_grid else history
        c_grid = _tolist(current) if c_is_grid else current
        reference = c_grid if c_is_grid else h_grid
        h, w = _grid_size(reference)
        mask: list[list[bool]] = []
        weights: list[list[float]] = []
        valid_count = 0
        for y in range(h):
            mask_row: list[bool] = []
            weight_row: list[float] = []
            for x in range(w):
                if h_is_grid:
                    a = h_grid[y][x]
                else:
                    a = history
                if c_is_grid:
                    b = c_grid[y][x]
                else:
                    b = current
                if not (_is_number(a) and _is_number(b)):
                    ok = False
                    delta = None
                else:
                    delta = abs(float(a) - float(b))
                    limit = abs_tol + rel_tol * max(
                        abs(float(a)), abs(float(b)), 1e-12
                    )
                    ok = delta <= limit
                mask_row.append(bool(ok))
                weight_row.append(1.0 if ok else 0.0)
                valid_count += int(ok)
            mask.append(mask_row)
            weights.append(weight_row)
        total = max(1, h * w)
        all_valid = valid_count == total
        return Report(
            all_valid,
            ok=True,
            valid=all_valid,
            checked=True,
            mask=mask,
            weights=weights,
            valid_count=valid_count,
            total=total,
            valid_ratio=valid_count / total,
            reason="depth_checked",
        )

    if not _is_number(history) or not _is_number(current):
        return Report(
            False,
            ok=False,
            valid=False,
            checked=False,
            reason="non_numeric_depth",
            weight=0.0,
            delta=None,
        )

    a, b = float(history), float(current)
    delta = abs(a - b)
    limit = abs_tol + rel_tol * max(abs(a), abs(b), 1e-12)
    valid = delta <= limit
    return Report(
        valid,
        ok=True,
        valid=valid,
        accepted=valid,
        rejected=not valid,
        checked=True,
        reason="depth_match" if valid else "depth_mismatch",
        history_depth=a,
        current_depth=b,
        delta=delta,
        threshold=limit,
        weight=1.0 if valid else 0.0,
        history_weight=1.0 if valid else 0.0,
    )


# ---------------------------------------------------------------------------
# 修复 4：深度/法线感知双边上采样
# ---------------------------------------------------------------------------


def _normal_weight(a: Any, b: Any, sigma: float = 0.15) -> float:
    if not (_is_sequence(a) and _is_sequence(b)):
        return 1.0
    av = [float(x) for x in _tolist(a)]
    bv = [float(x) for x in _tolist(b)]
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av)) or 1.0
    nb = math.sqrt(sum(y * y for y in bv)) or 1.0
    dot = _clamp(dot / (na * nb), -1.0, 1.0)
    return math.exp(-((1.0 - dot) ** 2) / max(1e-12, 2.0 * sigma * sigma))


def bilateral_upsample(*args: Any, **kwargs: Any) -> Any:
    """把半分辨率追踪/颜色结果做联合双边上采样。

    常用位置形式：``bilateral_upsample(low, full_depth)``；也支持
    ``bilateral_upsample(low, low_depth, full_depth, normals=...)``。
    深度差异会显著降低邻域权重，从而避免跨物体边缘的长条/锯齿。
    """
    options, positional = _merge_args(args, kwargs)
    low = _pick(options, "low", "values", "source", "color", "input", "data")
    if low is None and positional:
        low = positional[0]

    depth = _pick(options, "depth", "full_depth", "guide_depth", "depth_buffer")
    low_depth = _pick(options, "low_depth", "half_depth")
    normals = _pick(options, "normals", "normal_buffer", "full_normals")
    target_size = _pick(options, "target_size", "size", "output_size")

    grids = [a for a in positional[1:] if _is_grid(a)]
    if depth is None and grids:
        # 两个网格且形状与 low 相同则视为 low_depth，否则视为全分辨率深度。
        if len(grids) == 1:
            if _shape(grids[0])[:2] == _shape(low)[:2]:
                low_depth = grids[0]
            else:
                depth = grids[0]
        else:
            grids_sorted = sorted(grids, key=lambda g: _shape(g)[0] * _shape(g)[1])
            if low_depth is None:
                low_depth = grids_sorted[0]
            if depth is None:
                depth = grids_sorted[-1]

    if target_size is None:
        for candidate in positional[1:]:
            if _is_point(candidate) and all(
                float(x).is_integer() for x in _tolist(candidate)[:2]
            ):
                target_size = (int(candidate[1]), int(candidate[0]))
                break

    if low is None:
        return [[0.0]]
    if _is_number(low):
        return float(low)
    if not _is_grid(low):
        # 一维输入按一行数据处理。
        if _is_sequence(low):
            low = [_tolist(low)]
        else:
            return low

    prefer_numpy = _is_ndarray(low) or _is_ndarray(depth) or _is_ndarray(normals)
    low = _tolist(low)
    depth = _tolist(depth) if depth is not None else None
    low_depth = _tolist(low_depth) if low_depth is not None else None
    normals = _tolist(normals) if normals is not None else None

    lh, lw = _grid_size(low)
    if target_size is not None:
        th, tw = target_size
        fh, fw = int(th), int(tw)
    elif depth is not None and _is_grid(depth):
        fh, fw = _grid_size(depth)
    else:
        scale = int(_pick(options, "scale", "factor", default=2))
        fh, fw = lh * scale, lw * scale

    depth_sigma = float(
        _pick(options, "depth_sigma", "sigma_depth", default=0.05)
    )
    depth_threshold = _pick(options, "depth_threshold", default=None)
    normal_sigma = float(_pick(options, "normal_sigma", default=0.15))

    if low_depth is None and depth is not None:
        dh, dw = _grid_size(depth)
        if (dh, dw) == (lh, lw):
            low_depth = depth
            depth = None

    out: list[list[Any]] = [[0.0 for _ in range(fw)] for _ in range(fh)]
    for fy in range(fh):
        for fx in range(fw):
            u = (fx + 0.5) * lw / fw - 0.5
            v = (fy + 0.5) * lh / fh - 0.5
            x0 = int(math.floor(u))
            y0 = int(math.floor(v))
            tx = _clamp(u - x0, 0.0, 1.0)
            ty = _clamp(v - y0, 0.0, 1.0)
            spatial = (
                (1.0 - tx) * (1.0 - ty),
                tx * (1.0 - ty),
                (1.0 - tx) * ty,
                tx * ty,
            )
            full_z = depth[fy][fx] if depth is not None else None
            full_n = None
            if normals is not None:
                full_n = normals[fy][fx] if _grid_size(normals)[:2] == (fh, fw) else None

            acc: Any = 0.0
            wsum = 0.0
            nearest = low[int(_clamp(y0, 0, lh - 1))][int(_clamp(x0, 0, lw - 1))]
            for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                ly = int(_clamp(y0 + dy, 0, lh - 1))
                lx = int(_clamp(x0 + dx, 0, lw - 1))
                w = float(spatial[k])
                if w <= 1e-12:
                    continue

                if low_depth is not None and full_z is not None:
                    neighbor_z = low_depth[ly][lx]
                    dz = abs(float(neighbor_z) - float(full_z))
                    if depth_threshold is not None:
                        wz = max(0.0, 1.0 - dz / max(1e-12, float(depth_threshold)))
                    else:
                        sigma = max(1e-8, depth_sigma)
                        wz = math.exp(-(dz * dz) / (2.0 * sigma * sigma))
                    w *= wz
                elif depth is not None and full_z is not None:
                    # 低分辨率纹素中心在全分辨率深度图上的对应位置。
                    gu = ((lx + 0.5) / lw)
                    gv = ((ly + 0.5) / lh)
                    neighbor_z = _sample_grid_uv(depth, gu, gv)
                    if _is_number(neighbor_z):
                        dz = abs(float(neighbor_z) - float(full_z))
                        sigma = max(1e-8, depth_sigma)
                        w *= math.exp(-(dz * dz) / (2.0 * sigma * sigma))

                if normals is not None and full_n is not None:
                    if _grid_size(normals)[:2] == (lh, lw):
                        neighbor_n = normals[ly][lx]
                    else:
                        neighbor_n = _sample_grid_uv(
                            normals, (lx + 0.5) / lw, (ly + 0.5) / lh
                        )
                    w *= _normal_weight(full_n, neighbor_n, normal_sigma)

                if w <= 1e-12:
                    continue
                value = low[ly][lx]
                if isinstance(acc, list) or _is_sequence(value):
                    if not isinstance(acc, list):
                        acc = [0.0 for _ in range(len(value))]
                    for c, component in enumerate(value):
                        acc[c] += w * float(component)
                else:
                    acc += w * float(value)
                wsum += w

            if wsum > 1e-12:
                if isinstance(acc, list):
                    out[fy][fx] = [component / wsum for component in acc]
                else:
                    out[fy][fx] = acc / wsum
            else:
                out[fy][fx] = nearest

    if prefer_numpy and _np is not None:
        return _np.array(out)
    return out


# ---------------------------------------------------------------------------
# 修复 5：帧号和矩阵同帧捕获
# ---------------------------------------------------------------------------


def _matrix_from_snapshot(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, dict):
        frame = _pick(
            snapshot, "frame", "frame_index", "frame_id", "index", "time", "id"
        )
        view = _pick(
            snapshot,
            "view",
            "view_matrix",
            "current_view",
            "main_view",
            "camera_matrix",
            "matrix",
            default=snapshot if "frame" not in snapshot and not any(
                k in snapshot
                for k in ("projection", "proj", "view_proj", "view_projection", "vp", "matrix")
            )
            else None,
        )
        projection = _pick(snapshot, "projection", "proj", "projection_matrix")
        view_proj = _pick(snapshot, "view_proj", "view_projection", "vp")
        position = _pick(snapshot, "position", "camera_position", "origin", "eye")
        return {
            "frame": frame,
            "view": _copy_matrix(view),
            "projection": _copy_matrix(projection),
            "view_proj": _copy_matrix(view_proj),
            "position": list(position) if _is_sequence(position) else position,
        }
    if _is_sequence(snapshot) and not _is_point(snapshot):
        return {"frame": None, "view": _copy_matrix(snapshot), "projection": None, "view_proj": None, "position": None}
    return {"frame": None, "view": None, "projection": None, "view_proj": None, "position": None}


def same_frame_matrix(*args: Any, **kwargs: Any) -> Report:
    """同帧捕获主视图/投影矩阵，或校验两个快照是否同帧。"""
    options, positional = _merge_args(args, kwargs)
    raw_dicts = [a for a in args if isinstance(a, dict)]

    current: Any = None
    async_snapshot: Any = None
    if len(raw_dicts) >= 2:
        # 两个位置字典是最明确的快照比较形式，优先于 kwargs/合并键推断。
        current = raw_dicts[0]
        async_snapshot = raw_dicts[1]
    else:
        current = _pick(
            options,
            "current_snapshot",
            "main_snapshot",
            "camera_snapshot",
            "main",
            "current",
            "camera",
            "view",
        )
        async_snapshot = _pick(
            options,
            "trace_snapshot",
            "ssr_snapshot",
            "reflection_snapshot",
            "render_snapshot",
            "secondary_snapshot",
            "batch_snapshot",
            "async_snapshot",
            "trace_batch",
            "batch",
            "async",
        )
        if current is None and raw_dicts:
            current = raw_dicts[0]

    if current is not None and async_snapshot is not None:
        a = _matrix_from_snapshot(current)
        b = _matrix_from_snapshot(async_snapshot)
        fa = a["frame"]
        fb = b["frame"]
        lag = None
        frame_aligned = fa is not None and fb is not None and int(fa) == int(fb)
        if fa is not None and fb is not None:
            lag = int(fb) - int(fa)

        ma = a["view"] or a["view_proj"]
        mb = b["view"] or b["view_proj"]
        matrix_match = True
        if ma is not None and mb is not None:
            matrix_match = _matrix_digest(ma) == _matrix_digest(mb)
        aligned = frame_aligned and matrix_match
        strict = bool(_pick(options, "strict", default=True))
        ok = aligned if strict else lag is not None and abs(lag) <= int(
            _pick(options, "max_lag", default=0)
        )
        return Report(
            ok,
            ok=ok,
            aligned=aligned,
            source_aligned=aligned,
            frame_aligned=frame_aligned,
            matrix_match=matrix_match,
            current_frame=fa,
            camera_frame=fa,
            main_frame=fa,
            trace_frame=fb,
            async_frame=fb,
            shadow_frame=fb,
            batch_frame=fb,
            lag_frames=abs(lag) if lag is not None else None,
            source_lag_frames=abs(lag) if lag is not None else None,
            signed_lag_frames=lag,
            reason="same_frame" if aligned else "frame_or_matrix_mismatch",
        )

    # 捕获模式：same_frame_matrix(frame, view=None, projection=None)，
    # 也宽容 same_frame_matrix(view, projection, frame) 的矩阵优先写法。
    frame = _pick(options, "frame", "frame_index", "frame_id")
    view = _pick(
        options,
        "view",
        "view_matrix",
        "main_view",
        "camera_matrix",
        "matrix",
    )
    projection = _pick(options, "projection", "proj", "projection_matrix")
    positional_matrices = [p for p in positional if _is_grid(p)]
    positional_numbers = [p for p in positional if _is_number(p)]

    if frame is None and positional_numbers:
        frame = int(positional_numbers[0])
    if view is None and positional:
        if _is_number(positional[0]) and len(positional_matrices) >= 1:
            view = positional_matrices[0]
            if projection is None and len(positional_matrices) >= 2:
                projection = positional_matrices[1]
        elif positional_matrices:
            view = positional_matrices[0]
            if projection is None and len(positional_matrices) >= 2:
                projection = positional_matrices[1]
            if frame is None and len(positional_numbers) >= 1:
                frame = int(positional_numbers[-1])

    frame = int(frame) if _is_number(frame) else 0
    view_m = _copy_matrix(view) or identity_matrix()
    proj_m = _copy_matrix(projection)
    vp = multiply_matrices(proj_m, view_m) if proj_m is not None else None
    token = _stable_digest(
        {"frame": frame, "view": view_m, "projection": proj_m, "view_proj": vp}
    )[:16]
    return Report(
        True,
        ok=True,
        aligned=True,
        source_aligned=True,
        captured=True,
        immutable=True,
        frame=frame,
        frame_index=frame,
        camera_frame=frame,
        main_frame=frame,
        view_frame=frame,
        trace_frame=frame,
        async_frame=frame,
        batch_frame=frame,
        lag_frames=0,
        source_lag_frames=0,
        view=view_m,
        projection=proj_m,
        view_proj=vp,
        matrices={"view": view_m, "projection": proj_m, "view_proj": vp},
        token=token,
        matrix_token=token,
        reason="captured_same_frame",
    )


# ---------------------------------------------------------------------------
# 厚度、命中深度一致性与步进预算
# ---------------------------------------------------------------------------


def thickness(*args: Any, **kwargs: Any) -> ScalarReport:
    """返回非负深度厚度。

    位置形式 ``thickness(front_depth, back_depth)`` 与常见 SSR 约定一致：
    ``max(0, back-front)``。只传一个数值时把它视为厚度阈值。
    """
    options, positional = _merge_args(args, kwargs)
    front = _pick(options, "front", "front_depth", "ray_depth", default=None)
    back = _pick(options, "back", "back_depth", "scene_depth", "hit_depth", default=None)
    if front is None and positional:
        front = positional[0]
    if back is None and len(positional) >= 2:
        back = positional[1]

    if back is None:
        value = abs(float(_number(front, 0.0)))
        return ScalarReport(value, {"thickness": value, "limit": value})

    f, b = float(_number(front, 0.0)), float(_number(back, 0.0))
    value = max(0.0, b - f)
    return ScalarReport(
        value,
        {
            "thickness": value,
            "delta": value,
            "signed_delta": b - f,
            "absolute_delta": abs(b - f),
            "front": f,
            "back": b,
        },
    )


def hit_validate(*args: Any, **kwargs: Any) -> Report:
    """验证 SSR 命中是否在屏内且与深度缓冲一致。"""
    options, positional = _merge_args(args, kwargs)
    raw_dicts = [a for a in args if isinstance(a, dict)]
    hit = _pick(options, "hit", "hit_result", "candidate", default=None)
    depth_buffer = _pick(options, "depth_buffer", "z_buffer", default=None)
    depth_option = _pick(options, "depth", default=None)
    if depth_buffer is None and _is_grid(depth_option):
        depth_buffer = depth_option
    uv = _pick(options, "uv", "hit_uv", "screen_uv", default=None)
    hit_depth = _pick(
        options, "hit_depth", "candidate_depth", "ray_depth", "z", default=None
    )
    scene_depth = _pick(
        options, "scene_depth", "stored_depth", "depth_value", default=None
    )
    if scene_depth is None and _is_number(depth_option):
        scene_depth = depth_option

    grid_args = [p for p in positional if _is_grid(p)]
    point_args = [p for p in positional if _is_point(p)]
    number_args = [p for p in positional if _is_number(p)]
    if depth_buffer is None and grid_args:
        depth_buffer = grid_args[0]
    if uv is None and point_args:
        uv = point_args[0]
    if scene_depth is None and number_args:
        scene_depth = number_args[0]

    if hit is None and raw_dicts:
        hit = raw_dicts[0]
    if uv is None and len(point_args) >= 1:
        uv = point_args[0]
    if hit_depth is None and len(number_args) >= 2:
        hit_depth = number_args[0]
        if scene_depth is None:
            scene_depth = number_args[1]
    elif hit_depth is None and len(number_args) >= 1 and scene_depth is not None:
        # 只剩一个未命名数值且已经有场景深度时，把它视为候选射线深度。
        hit_depth = number_args[0]

    reasons: list[str] = []
    candidate_uv = uv
    candidate_z = hit_depth
    if isinstance(hit, dict):
        candidate_uv = candidate_uv or hit.get("uv") or hit.get("hit_uv")
        candidate_z = candidate_z if candidate_z is not None else _pick(
            hit, "hit_depth", "depth", "ray_depth", "z"
        )
        if hit.get("hit") is False or hit.get("valid") is False:
            reasons.append("reported_miss")
    elif _is_point(hit) and candidate_uv is None:
        candidate_uv = hit

    if candidate_uv is not None:
        offscreen = offscreen_reject(candidate_uv, **options)
        if bool(offscreen):
            reasons.append("offscreen")

    if depth_buffer is not None and candidate_uv is not None and scene_depth is None:
        uv_for_sample = _point_to_uv(candidate_uv, options)
        if uv_for_sample is not None:
            scene_depth = _sample_grid_uv(depth_buffer, *uv_for_sample)

    thickness_limit = float(
        _pick(options, "thickness", "depth_thickness", default=0.05)
    )
    bias = float(_pick(options, "bias", "depth_bias", default=1e-4))
    delta = None
    limit = thickness_limit
    if _is_number(candidate_z) and _is_number(scene_depth):
        delta = abs(float(candidate_z) - float(scene_depth))
        limit = thickness_limit + bias
        if delta > limit:
            reasons.append("depth_inconsistent")
    elif options.get("require_depth", False):
        reasons.append("depth_missing")

    valid = not reasons
    return Report(
        valid,
        ok=valid,
        valid=valid,
        accepted=valid,
        rejected=not valid,
        reasons=reasons,
        reason=reasons[0] if reasons else "valid",
        uv=candidate_uv,
        hit_depth=candidate_z,
        scene_depth=scene_depth,
        depth_delta=delta,
        thickness=thickness_limit,
        threshold=limit,
    )


def step_budget(*args: Any, **kwargs: Any) -> IntReport:
    """根据距离、最大步数和分辨率比例返回可追踪步数。"""
    options, positional = _merge_args(args, kwargs)
    max_steps = int(_pick(options, "max_steps", "maximum", "budget", default=64))
    distance = _pick(options, "distance", "ray_distance", "length", default=None)
    density = float(_pick(options, "density", "step_density", "base", default=8.0))
    roughness = _pick(options, "roughness", default=0.0)
    scale = _pick(options, "resolution_scale", "scale", default=None)

    if positional and _is_number(positional[0]):
        first = float(positional[0])
        if distance is None and (len(positional) >= 2 or first > 4.0):
            distance = first
        else:
            max_steps = int(first)
    if len(positional) >= 2 and _is_number(positional[1]):
        if distance is None:
            distance = float(positional[0])
            density = float(positional[1])
        elif scale is None:
            density = float(positional[1])

    if _is_number(roughness) and scale is None:
        scale = roughness_layered(roughness).get("step_scale", 1.0)
    scale = float(1.0 if scale is None else scale)
    if distance is None:
        steps = max(1, max_steps)
    else:
        steps = int(math.ceil(abs(float(distance)) * density * max(0.0, scale)))
        steps = max(1, min(max_steps, steps))

    return IntReport(
        steps,
        {
            "steps": steps,
            "budget": steps,
            "max_steps": max_steps,
            "distance": distance,
            "density": density,
            "scale": scale,
        },
    )


# ---------------------------------------------------------------------------
# 模糊
# ---------------------------------------------------------------------------


def _gaussian_kernel(radius: int, sigma: float | None = None) -> list[float]:
    radius = max(0, int(radius))
    sigma = max(1e-6, float(sigma if sigma is not None else max(0.75, radius / 2.0)))
    weights = [
        math.exp(-(i * i) / (2.0 * sigma * sigma))
        for i in range(-radius, radius + 1)
    ]
    total = sum(weights)
    return [w / total for w in weights]


def _blur_1d(values: list[Any], radius: int, sigma: float | None) -> list[Any]:
    kernel = _gaussian_kernel(radius, sigma)
    n = len(values)
    out: list[Any] = []
    for i in range(n):
        acc: Any = 0.0
        for k, weight in enumerate(kernel, start=-radius):
            j = int(_clamp(i + k, 0, n - 1))
            value = values[j]
            if isinstance(acc, list) or _is_sequence(value):
                if not isinstance(acc, list):
                    acc = [0.0 for _ in value]
                for c, component in enumerate(value):
                    acc[c] += weight * float(component)
            else:
                acc += weight * float(value)
        out.append(acc)
    return out


def blur(*args: Any, **kwargs: Any) -> Any:
    """对一维序列、二维标量网格或 HxWxC 颜色做边缘钳制高斯模糊。"""
    options, positional = _merge_args(args, kwargs)
    data = _pick(options, "data", "values", "image", "color", default=None)
    if data is None and positional:
        data = positional[0]
    radius = _pick(options, "radius", "blur_radius", default=None)
    sigma = _pick(options, "sigma", "blur_sigma", default=None)
    roughness = _pick(options, "roughness", default=None)
    if radius is None and len(positional) >= 2 and _is_number(positional[1]):
        radius = int(positional[1])
    if radius is None:
        if _is_number(roughness):
            plan = roughness_layered(float(roughness))
            radius = plan["blur_radius"]
            sigma = sigma if sigma is not None else plan["blur_sigma"]
        else:
            radius = 1
    radius = max(0, int(radius))

    prefer_numpy = _is_ndarray(data)
    if data is None:
        return 0.0
    if _is_number(data):
        return float(data)
    if not _is_sequence(data):
        return data
    data = _tolist(data)

    if _is_grid(data):
        h, w = _grid_size(data)
        horizontal = [_blur_1d(list(row), radius, sigma) for row in data[:h]]
        # 转置后再做一次一维模糊，形成可分离二维核。
        transposed = [
            [horizontal[y][x] for y in range(h)] for x in range(w)
        ]
        blurred_t = [_blur_1d(row, radius, sigma) for row in transposed]
        out = [[blurred_t[x][y] for x in range(w)] for y in range(h)]
    else:
        out = _blur_1d(list(data), radius, sigma)

    if prefer_numpy and _np is not None:
        return _np.array(out)
    return out


# ---------------------------------------------------------------------------
# 修复 6：屏幕空间追踪与深度一致性
# ---------------------------------------------------------------------------


def trace(*args: Any, **kwargs: Any) -> Report:
    """在深度缓冲上执行线性/步进式 SSR，并落实六项修复。

    常用关键字：``origin_uv``、``direction_uv`` 或 ``end_uv``、``origin_depth``、
    ``depth``、``roughness``、``frame_packet``、``max_steps``。
    """
    options, positional = _merge_args(args, kwargs)

    depth = _pick(options, "depth", "depth_buffer", "z_buffer", "scene_depth")
    if depth is None:
        grids = [p for p in positional if _is_grid(p)]
        if grids:
            depth = grids[0]

    origin = _pick(
        options, "origin_uv", "start_uv", "uv", "screen_uv", "origin", "start"
    )
    direction = _pick(
        options, "direction_uv", "ray_uv", "direction", "ray", "delta_uv"
    )
    end_uv = _pick(options, "end_uv", "target_uv", "end", "target")
    origin_depth = _pick(
        options, "origin_depth", "start_depth", "ray_origin_depth", "z", default=0.0
    )
    ray_depth_slope = float(
        _pick(options, "depth_slope", "ray_depth_slope", "dz", default=0.0)
    )
    roughness = float(_pick(options, "roughness", default=0.0))
    max_steps = int(_pick(options, "max_steps", default=64))
    step_size = _pick(options, "step_size", "step_size_uv", default=None)
    thickness_limit = float(_pick(options, "thickness", default=0.05))
    bias = float(_pick(options, "bias", default=1e-4))
    packet = _pick(options, "frame_packet", "matrix_packet", "same_frame")

    points = [p for p in positional if _is_point(p)]
    if origin is None and points:
        origin = points[0]
    if direction is None and len(points) >= 2:
        direction = points[1]

    if packet is not None:
        packet_report = packet if isinstance(packet, Report) else same_frame_matrix(packet)
        if not bool(packet_report.get("ok", bool(packet_report))):
            return Report(
                False,
                hit=False,
                ok=False,
                valid=False,
                reason="matrix_frame_mismatch",
                frame_packet=dict(packet_report),
                steps=0,
            )

    plan = roughness_layered(roughness)
    if options.get("respect_layers", True) and not plan["specular_trace"] and not options.get("force_trace", False):
        fallback_report = fallback("roughness_layer", roughness=roughness)
        return Report(
            False,
            hit=False,
            ok=True,
            valid=False,
            accepted=False,
            reason="roughness_layer_fallback",
            layer=plan.to_dict(),
            fallback=fallback_report,
            steps=0,
        )

    if origin is None or not _is_point(origin):
        return Report(
            False,
            hit=False,
            ok=True,
            valid=False,
            reason="no_ray",
            demo=True,
            steps=0,
            layer=plan.to_dict(),
        )

    uv0 = _point_to_uv(origin, options)
    if uv0 is None:
        return Report(False, hit=False, ok=False, reason="invalid_origin", steps=0)
    u0, v0 = uv0

    if end_uv is not None and _is_point(end_uv):
        eu, ev = _point_to_uv(end_uv, options)
        du, dv = eu - u0, ev - v0
        parametric = True
        steps = int(step_budget(max_steps, roughness=roughness))
    elif direction is not None and _is_point(direction):
        du, dv = float(direction[0]), float(direction[1])
        parametric = step_size is None
        if step_size is not None:
            du *= float(step_size)
            dv *= float(step_size)
        steps = int(step_budget(max_steps, roughness=roughness))
    else:
        du = dv = 0.0
        parametric = True
        steps = 1

    z0 = float(_number(origin_depth, 0.0))
    last_uv = (u0, v0)
    last_z = z0

    for i in range(1, max(1, steps) + 1):
        if parametric:
            t = i / float(steps)
            u = u0 + du * t
            v = v0 + dv * t
            ray_z = z0 + ray_depth_slope * t
        else:
            u = u0 + du * i
            v = v0 + dv * i
            ray_z = z0 + ray_depth_slope * i

        last_uv = (u, v)
        last_z = ray_z

        # 修复 1：任何一步出屏都立即失败，绝不钳到纹理边缘继续报命中。
        reject = offscreen_reject((u, v), **options)
        if bool(reject):
            return Report(
                False,
                hit=False,
                ok=True,
                valid=False,
                rejected=True,
                offscreen=True,
                reason="offscreen",
                uv=(u, v),
                step=i,
                steps=i,
                layer=plan.to_dict(),
            )

        scene_z = _sample_grid_uv(depth, u, v) if depth is not None else None
        candidate = {
            "uv": (u, v),
            "hit_depth": ray_z,
            "depth": scene_z,
        }
        check = hit_validate(
            candidate,
            depth_buffer=depth,
            thickness=thickness_limit,
            bias=bias,
            require_depth=True,
        )
        # 对线性视深，射线从相机向前走；额外要求射线没有落到场景表面之后太远。
        depth_crossed = True
        if _is_number(scene_z):
            depth_crossed = ray_z + bias >= float(scene_z) - thickness_limit
        if bool(check) and depth_crossed:
            return Report(
                True,
                hit=True,
                ok=True,
                valid=True,
                reason="hit",
                uv=(u, v),
                hit_uv=(u, v),
                hit_depth=ray_z,
                scene_depth=scene_z,
                depth_delta=check.get("depth_delta"),
                step=i,
                steps=i,
                layer=plan.to_dict(),
                validation=dict(check),
            )

    return Report(
        False,
        hit=False,
        ok=True,
        valid=True,
        reason="no_hit",
        uv=last_uv,
        hit_depth=last_z,
        steps=steps,
        layer=plan.to_dict(),
    )


# ---------------------------------------------------------------------------
# 确定性、指标、审计、校验与回退
# ---------------------------------------------------------------------------


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """生成稳定的伪随机序列；相同输入始终得到逐位相同的报告。"""
    options, positional = _merge_args(args, kwargs)
    source = _pick(options, "data", "values", "points", default=None)
    seed = _pick(options, "seed", default=None)
    if seed is None and positional and _is_number(positional[0]):
        seed = int(positional[0])
    if seed is None and source is not None:
        seed = int(_stable_digest(_tolist(source))[:8], 16) & 0xFFFF
    if seed is None:
        seed = 11

    count = int(_pick(options, "count", "n", default=8))
    runs = int(_pick(options, "runs", default=2))
    values = [_hash01(seed * 1000003 + i * 31 + 17) for i in range(max(1, count))]
    run_values = [list(values) for _ in range(max(1, runs))]
    equal = all(v == values for v in run_values)
    return {
        "seed": seed,
        "count": len(values),
        "runs": runs,
        "values": values,
        "run_values": run_values,
        "equal": equal,
        "deterministic": True,
        "digest": _stable_digest(values),
    }


def validation(*args: Any, **kwargs: Any) -> Report:
    """校验 SSR 配置/数据；标量冒烟输入返回 ``ok=True``。"""
    options, positional = _merge_args(args, kwargs)
    errors: list[str] = []
    warnings: list[str] = []

    roughness = _pick(options, "roughness", default=None)
    if _is_number(roughness) and not (0.0 <= float(roughness) <= 1.0):
        errors.append("roughness:out-of-range")

    frame = _pick(options, "frame", default=None)
    if _is_number(frame) and int(frame) < 0:
        errors.append("frame:negative")

    for matrix_key in ("view", "projection", "view_proj"):
        matrix = _pick(options, matrix_key, default=None)
        if matrix is not None:
            copied = _copy_matrix(matrix)
            if copied is None or len(_shape(copied)) != 2:
                errors.append(f"{matrix_key}:invalid-matrix")
            elif matrix_key != "view_proj" and (len(copied) != 4 or any(len(r) != 4 for r in copied)):
                warnings.append(f"{matrix_key}:expected-4x4")

    depth = _pick(options, "depth", "depth_buffer", default=None)
    if depth is not None:
        numbers = _flatten_numbers(depth)
        if any(not math.isfinite(x) for x in numbers):
            errors.append("depth:non-finite")
        elif any(x < 0.0 for x in numbers):
            warnings.append("depth:negative-values")

    for arg in positional:
        numbers = _flatten_numbers(arg)
        if numbers and any(not math.isfinite(x) for x in numbers):
            errors.append("input:non-finite")

    if not errors and not positional and not options:
        warnings.append("empty-validation")

    ok = not errors
    return Report(
        ok,
        ok=ok,
        valid=ok,
        errors=errors,
        warnings=warnings,
        error_count=len(errors),
        warning_count=len(warnings),
    )


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """生成可复现的审计摘要和基础计数。"""
    options, positional = _merge_args(args, kwargs)
    events = _pick(options, "events", default=None)
    if events is None:
        events = positional[0] if positional else None
    if isinstance(events, (list, tuple)):
        event_count = len(events)
        event_data = list(events)
    elif isinstance(events, dict) and isinstance(events.get("events"), list):
        event_count = len(events["events"])
        event_data = events["events"]
    else:
        event_count = 1 if events is not None or args or kwargs else 0
        event_data = [events] if events is not None else []

    digest = _stable_digest({"args": _tolist(args), "kwargs": options})
    return {
        "ok": True,
        "event_count": event_count,
        "events": event_data,
        "digest": digest,
        "sha256": digest if len(digest) == 64 else _stable_digest(digest),
        "counts": {
            "events": event_count,
            "errors": 0,
            "warnings": 0,
            "offscreen": int(_pick(options, "offscreen_count", default=0) or 0),
            "depth_rejected": int(_pick(options, "depth_rejected_count", default=0) or 0),
        },
    }


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """SSR 不可信时的探针/环境/统一色回退。"""
    options, positional = _merge_args(args, kwargs)
    reason = _pick(options, "reason", default=None)
    if reason is None and positional and isinstance(positional[0], str):
        reason = positional[0]
    value = _pick(options, "color", "value", "intensity", default=None)
    if value is None and positional and _is_number(positional[0]):
        value = float(positional[0])
    if value is None and not positional:
        value = 0.0

    if _is_number(value):
        color = (float(value), float(value), float(value))
    elif _is_sequence(value):
        seq = [float(x) for x in _tolist(value)]
        color = tuple((seq + [0.0, 0.0, 0.0])[:3])
    else:
        color = (0.04, 0.04, 0.05)

    mode = str(_pick(options, "mode", default="uniform_probe"))
    return {
        "ok": True,
        "mode": mode,
        "source": _pick(options, "source", default="reflection_probe"),
        "probe": "environment",
        "reason": reason or "ssr_unavailable",
        "color": color,
        "value": value,
        "confidence": float(_pick(options, "confidence", default=0.0)),
        "history_weight": 0.0,
        "aliases": ("uniform", "probe", "environment", "sky"),
    }


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """计算 RMS 以及 SSR 管线常用比率指标。"""
    options, positional = _merge_args(args, kwargs)
    target = _pick(options, "values", "data", "result", default=None)
    if target is None and positional:
        target = positional[0]
    compare = _pick(options, "compare", "other", "smoothed", default=None)
    if compare is None and len(positional) >= 2:
        compare = positional[1]

    numbers = _flatten_numbers(target)
    rms = math.sqrt(sum(x * x for x in numbers) / len(numbers)) if numbers else 0.0
    mean = sum(numbers) / len(numbers) if numbers else 0.0
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers) if numbers else 0.0

    result = {
        "ok": True,
        "rms": rms if numbers else 1.0 if _is_number(target) and float(target) == 1.0 else 0.0,
        "mean": mean,
        "variance": variance,
        "stddev": math.sqrt(variance),
        "count": len(numbers),
        "hit_rate": _pick(options, "hit_rate", default=None),
        "offscreen_rate": _pick(options, "offscreen_rate", default=None),
        "depth_reject_rate": _pick(options, "depth_reject_rate", default=None),
        "stale_history_rate": _pick(options, "stale_history_rate", default=None),
        "aliasing_score": _pick(options, "aliasing_score", default=None),
    }
    # 契约/冒烟：metrics(1.0)["rms"] == 1.0。
    if not numbers and _is_number(target):
        result["rms"] = abs(float(target))

    if compare is not None:
        a = _flatten_numbers(target)
        b = _flatten_numbers(compare)
        n = min(len(a), len(b))
        if n:
            diffs = [a[i] - b[i] for i in range(n)]
            result["difference_rms"] = math.sqrt(sum(d * d for d in diffs) / n)
            result["jitter_before"] = math.sqrt(
                sum((a[i] - (a[i - 1] if i else a[i])) ** 2 for i in range(1, n))
                / max(1, n - 1)
            )
            result["jitter_after"] = math.sqrt(
                sum((b[i] - (b[i - 1] if i else b[i])) ** 2 for i in range(1, n))
                / max(1, n - 1)
            )
            if result["jitter_before"] > 0:
                result["residual_ratio"] = _clamp(
                    result["jitter_after"] / result["jitter_before"]
                )
    return result
