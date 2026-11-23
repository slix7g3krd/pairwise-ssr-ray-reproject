# pairwise-ssr-ray-reproject

## 领域

游戏、图形与多媒体

## 主题

屏幕空间反射追踪

## 说明

出屏判定、粗糙度分层与历史校验。

初始状态下 `graphics/ssr_ray_reproject.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
