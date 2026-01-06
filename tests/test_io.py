from utils.io import output_path_for_video
import os


def test_output_path_no_suffix(tmp_path):
    inp = "/data/videos/subj1.mp4"
    out = output_path_for_video(inp, str(tmp_path))
    assert os.path.basename(out) == "subj1.ply"


def test_output_path_with_suffix(tmp_path):
    inp = "/data/videos/scan-01.avi"
    out = output_path_for_video(inp, str(tmp_path), suffix="points")
    assert os.path.basename(out) == "scan-01_points.ply"
    assert out.startswith(str(tmp_path))
