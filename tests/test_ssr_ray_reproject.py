import pytest
import graphics.ssr_ray_reproject as mod

CASES = [
    ("offscreen_reject", (1,)),
    ("roughness_layered", (1,)),
    ("history_depth_check", (1,)),
    ("bilateral_upsample", (1,)),
    ("same_frame_matrix", (1,)),
    ("determinism", (1,)),
    ("trace", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("step_budget", (1,)),
    ("metrics", (1,)),
    ("hit_validate", (1,)),
    ("thickness", (1,)),
    ("blur", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
