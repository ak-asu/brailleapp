from pipeline.runtime import (
    should_run_onnx,
    should_hold_for_blur,
    with_guidance,
    is_new_frame,
)


def test_should_not_run_fixed_onnx_on_downscaled_frame():
    assert should_run_onnx(session=object(), counter=3, target_size=480) is False


def test_should_run_onnx_every_third_full_size_frame():
    assert should_run_onnx(session=object(), counter=3, target_size=640) is True
    assert should_run_onnx(session=object(), counter=4, target_size=640) is False
    assert should_run_onnx(session=None, counter=3, target_size=640) is False


def test_should_hold_for_very_blurry_frame_only():
    assert should_hold_for_blur(49.9) is True
    assert should_hold_for_blur(50.0) is False


def test_with_guidance_preserves_previous_result_values():
    last = {
        "annotated_frame": "abc",
        "text": "hello",
        "confidence": 0.8,
        "guidance": {"status": "ok", "message": "Good"},
    }
    guidance = {"status": "warn", "message": "Hold camera steady"}
    result = with_guidance(last, guidance)
    assert result["annotated_frame"] == "abc"
    assert result["text"] == "hello"
    assert result["confidence"] == 0.8
    assert result["guidance"] == guidance
    assert result is not last


def test_is_new_frame_rejects_duplicate_or_missing_timestamps():
    assert is_new_frame({"ts": 2}, last_ts=1) is True
    assert is_new_frame({"ts": 1}, last_ts=1) is False
    assert is_new_frame({}, last_ts=1) is True
