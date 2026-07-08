import json
import os
import cv2
import numpy as np
from typing import Dict

_FRAME_EXTS = (".jpg", ".jpeg", ".png")


def _list_frame_files(frames_dir: str) -> list:
    """Sorted list of frame image paths in a GT frames directory."""
    return sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(_FRAME_EXTS)
    )


def load_intrinsics(path: str) -> Dict[str, float]:
    """
    Loads intrinsics.npy with shape (N, 4) where each row is [fx, fy, cx, cy].
    Returns a dict with keys fx, fy, cx, cy from the first row.
    """
    data = np.load(path).astype(np.float64)
    if data.ndim == 2:
        row = data[0]
    elif data.ndim == 1:
        row = data
    else:
        raise ValueError(f"Unexpected intrinsics shape {data.shape} in {path}")
    return {"fx": float(row[0]), "fy": float(row[1]), "cx": float(row[2]), "cy": float(row[3])}


def load_gt_frame_count_synthetic(gt_source: str) -> int:
    """
    Returns the total GT frame count.

    gt_source may be either a GT video file (frame count read from the
    container) or a directory of extracted GT frames (count of image files,
    e.g. Synthetic_processed/{scene}/{clip}/RGB{n}/ or Real_Raw/{id}/FRAMES/).

    GT clips are NOT all 300 frames — they range from ~150 to 300 depending
    on the scene, so we always read the actual count from the source.
    """
    if os.path.isdir(gt_source):
        n = len(_list_frame_files(gt_source))
        if n <= 0:
            raise RuntimeError(f"No frame images found in {gt_source}")
        return n
    cap = cv2.VideoCapture(gt_source)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Cannot read frame count from {gt_source}")
    return n


def extract_gt_frame_at(gt_source: str, frame_idx: int) -> np.ndarray:
    """
    Extract a single GT frame by index (0-based).
    gt_source may be a video file or a directory of extracted frames.
    Returns a BGR numpy array.
    """
    if os.path.isdir(gt_source):
        files = _list_frame_files(gt_source)
        if not (0 <= frame_idx < len(files)):
            raise RuntimeError(f"Frame {frame_idx} out of range for {gt_source}")
        frame = cv2.imread(files[frame_idx])
        if frame is None:
            raise RuntimeError(f"Cannot read {files[frame_idx]}")
        return frame
    cap = cv2.VideoCapture(gt_source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(
            f"Cannot read frame {frame_idx} from {gt_source}"
        )
    return frame


def extract_gt_frames_batch(gt_source: str, indices: list) -> list:
    """
    Extract multiple GT frames in one pass.
    gt_source may be a video file (single open/close) or a directory of
    extracted frames. indices: sorted list of 0-based frame indices.
    Returns a list of BGR numpy arrays in the same order as indices.
    """
    if not indices:
        return []
    if os.path.isdir(gt_source):
        files = _list_frame_files(gt_source)
        frames = []
        for idx in indices:
            if not (0 <= idx < len(files)):
                raise RuntimeError(f"Frame {idx} out of range for {gt_source}")
            frame = cv2.imread(files[idx])
            if frame is None:
                raise RuntimeError(f"Cannot read {files[idx]}")
            frames.append(frame)
        return frames
    cap = cv2.VideoCapture(gt_source)
    frames = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Cannot read frame {idx} from {gt_source}")
        frames.append(frame)
        prev_idx = idx
    cap.release()
    return frames


def load_intrinsics_json(path: str) -> Dict[str, float]:
    """
    Loads intrinsics from a Real GT JSON file.
    Expected format: {"camera": {"fx": ..., "fy": ..., "cx": ..., "cy": ..., ...}}
    Returns a dict with keys fx, fy, cx, cy.
    """
    with open(path) as f:
        data = json.load(f)
    cam = data.get("camera", data)   # handle both nested and flat formats
    return {
        "fx": float(cam["fx"]),
        "fy": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
    }


def load_gt_frame_count_real(timestamps_path: str) -> int:
    """
    Returns the total GT frame count for a Real clip by counting non-empty
    lines in timestamps.txt (one timestamp per GT frame).

    Legacy loader — the released dataset does not ship timestamps.txt.
    Use load_gt_frame_count_synthetic() on the clip's GT mp4 instead.
    """
    with open(timestamps_path) as f:
        return sum(1 for line in f if line.strip())


def load_intrinsics_npz(path: str) -> Dict[str, float]:
    """
    Loads intrinsics from a MapAnything npz shipped with each Real clip
    (Real_Raw/{id}/raw/{id}_mapanything.npz, key 'intrinsic').
    Accepts a (3,3) K matrix, a flat (4,) [fx, fy, cx, cy], or (N,4) rows.
    Returns a dict with keys fx, fy, cx, cy.
    """
    data = np.load(path)
    K = np.asarray(data["intrinsic"], dtype=np.float64)
    if K.ndim == 3:                      # (N, 3, 3) — take first
        K = K[0]
    if K.ndim == 2 and K.shape == (3, 3):
        return {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    if K.ndim == 2 and K.shape[1] == 4:  # (N, 4) rows
        K = K[0]
    if K.ndim == 1 and K.shape[0] >= 4:  # flat [fx, fy, cx, cy]
        return {"fx": float(K[0]), "fy": float(K[1]),
                "cx": float(K[2]), "cy": float(K[3])}
    raise ValueError(f"Unexpected intrinsic shape {K.shape} in {path}")


def extract_gt_frame0(mp4_path: str) -> np.ndarray:
    """
    Extract the first frame from a Real GT MP4 as a BGR numpy array.
    Used as the reference image for ReferenceFidelity.
    """
    cap = cv2.VideoCapture(mp4_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from {mp4_path}")
    return frame
