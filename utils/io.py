import os
from typing import Optional


def output_path_for_video(video_path: str, out_dir: str, suffix: Optional[str] = None) -> str:
    """Return a safe output path for a video file in out_dir.

    Example: video.mp4 -> out_dir/video.ply or out_dir/video_suffix.ply
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    if suffix:
        name = f"{base}_{suffix}.ply"
    else:
        name = f"{base}.ply"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)
