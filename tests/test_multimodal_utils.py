from src.utils import multimodal_utils


def test_img_tokens_336_p14():
    r = multimodal_utils.img_tokens(336, 336, patch=14)
    assert r["total_patch_tokens"] == 24 * 24  # 336/14=24


def test_img_tokens_higher_res():
    small = multimodal_utils.img_tokens(336, 336, 14)
    big = multimodal_utils.img_tokens(1024, 1024, 14)
    assert big["total_patch_tokens"] > small["total_patch_tokens"]


def test_vlm_cost_positive():
    r = multimodal_utils.vlm_cost(100, 500, "gpt4o")
    assert r["total_cost_usd"] > 0


def test_video_tokens_scales_with_duration():
    short = multimodal_utils.video_tokens(10, 1.0, 576)
    long = multimodal_utils.video_tokens(600, 1.0, 576)
    assert long["total_tokens"] > short["total_tokens"]


def test_video_tokens_scales_with_fps():
    low_fps = multimodal_utils.video_tokens(60, 0.5, 576)
    high_fps = multimodal_utils.video_tokens(60, 2.0, 576)
    assert high_fps["total_tokens"] > low_fps["total_tokens"]

