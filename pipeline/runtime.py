from typing import Any


def should_run_onnx(session: Any, counter: int, target_size: int) -> bool:
    """Run fixed-size YOLO ONNX only on every third full-size frame."""
    return session is not None and counter % 3 == 0 and target_size == 640


def should_hold_for_blur(blur_score: float, threshold: float = 50.0) -> bool:
    """Very blurry frames are skipped to avoid translating transient noise."""
    return blur_score < threshold


def with_guidance(last_result: dict, guidance: dict) -> dict:
    """Return the previous result payload with fresh guidance attached."""
    result = dict(last_result)
    result["guidance"] = guidance
    return result


def is_new_frame(frame_data: dict, last_ts: int | float | None) -> bool:
    """Reject duplicate component frames when the JS timestamp is available."""
    ts = frame_data.get("ts")
    return ts is None or ts != last_ts
