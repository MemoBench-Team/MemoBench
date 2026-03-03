"""
CLIP-based metrics:
  - SceneConsistency: mean consecutive CLIP cosine similarity across sampled frames
    within a given [start, end] range. Catches sudden scene identity shifts.
      - scene_consistency    : mean of sim(frame_i, frame_{i+1})
      - min_consecutive_sim  : minimum consecutive sim (detects sudden jumps)

  - RevisitMetrics: phase-aware OHR metrics using CLIP similarity vs frame_0.
      - revisit_recovery : (s_R - s_H) / (s_O - s_H + ε) — how much R phase
                           recovers toward O phase baseline after H phase drop
      - revisit_bridge   : sim(frame[h_start], frame[r_start]) — scene
                           continuity across the hidden interval boundary

  - PromptAlignment: mean CLIP cosine similarity between the text prompt and
    sampled frames. Measures instruction/prompt following quality.
"""

import numpy as np
import torch
from PIL import Image
import cv2


_model = None
_preprocess = None
_device = None


def _load_clip(device: str = None):
    global _model, _preprocess, _device
    if _model is not None:
        return _model, _preprocess, _device

    import open_clip

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = device

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _model = _model.to(_device).eval()
    return _model, _preprocess, _device


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def _embed(model, preprocess, device, img: np.ndarray) -> np.ndarray:
    t = preprocess(_bgr_to_pil(img)).unsqueeze(0).to(device)
    feat = model.encode_image(t)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().float().numpy()


def _phase_indices(start: int, end: int, n: int, total: int) -> list:
    """Uniformly spaced indices in [start, end], clamped to [0, total-1]."""
    start = max(0, min(start, total - 1))
    end   = max(start, min(end, total - 1))
    if n <= 1 or start == end:
        return [start]
    return [start + int(i * (end - start) / (n - 1)) for i in range(n)]


def scene_consistency_score(frames, n_sample: int = 9, device: str = None,
                             start: int = 0, end: int = None) -> dict:
    """
    Mean CLIP cosine similarity between consecutive sampled frames within
    [start, end]. Catches sudden scene identity shifts or jump cuts.

    For phase-aware OHR evaluation call with start/end set to the O or R phase
    boundaries to avoid penalising the intentional camera motion in H phase.

    Returns:
      - scene_consistency  : mean consecutive cosine similarity
      - min_consecutive_sim: minimum consecutive similarity (flags sudden jumps)
    """
    model, preprocess, device = _load_clip(device)

    N = frames.num_frames
    if end is None:
        end = N - 1

    indices = _phase_indices(start, end, n_sample, N)
    if len(indices) < 2:
        return {"scene_consistency": 1.0, "min_consecutive_sim": 1.0}

    embs = [_embed(model, preprocess, device, frames.get(idx)) for idx in indices]
    consecutive_sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(embs) - 1)]

    return {
        "scene_consistency":   round(float(np.mean(consecutive_sims)), 4),
        "min_consecutive_sim": round(float(np.min(consecutive_sims)),  4),
    }


def revisit_metrics(frames, h_start: int, r_start: int,
                    n_sample: int = 6, device: str = None) -> dict:
    """
    Phase-aware OHR metrics using CLIP similarity vs frame_0 as the anchor.

    Splits the video into three phases using the GT-derived keyframe boundaries:
      O phase: [0,       h_start]  — observer sees target
      H phase: [h_start, r_start]  — target is hidden (outside FOV)
      R phase: [r_start, N-1    ]  — target re-enters FOV

    Computes mean CLIP cosine similarity vs frame_0 for each phase:
      s_O, s_H, s_R

    RevisitRecovery:
      Measures how much the R phase recovered toward the O phase baseline
      after the H phase drop.
        recovery = clip((s_R - s_H) / (s_O - s_H + ε), 0, 1)
        1.0 → R phase fully returns to O phase similarity level
        0.0 → R phase stays at H phase level (no semantic recovery)

    RevisitBridge:
      CLIP cosine similarity between frame[h_start] and frame[r_start].
      Measures whether there is an abrupt scene jump at the H→R boundary.
      Low value → identity reset or jump cut across the hidden interval.

    Returns:
      - revisit_recovery : float in [0, 1]
      - revisit_bridge   : float in [0, 1]
      - s_O, s_H, s_R    : per-phase mean similarities (diagnostic)
    """
    model, preprocess, device = _load_clip(device)

    N = frames.num_frames
    h = max(1, min(h_start, N - 2))
    r = max(h + 1, min(r_start, N - 1))

    emb_0 = _embed(model, preprocess, device, frames.get(0))

    def phase_mean_sim(p_start, p_end):
        idxs = _phase_indices(p_start, p_end, n_sample, N)
        sims = [float(np.dot(emb_0, _embed(model, preprocess, device, frames.get(i))))
                for i in idxs]
        return float(np.mean(sims))

    s_O = phase_mean_sim(0, h)
    s_H = phase_mean_sim(h, r)
    s_R = phase_mean_sim(r, N - 1)

    # RevisitRecovery: how much of the H-phase drop was recovered in R phase
    drop = s_O - s_H
    if drop < 1e-4:
        # No meaningful drop in H phase — check R is at least as good as O
        recovery = 1.0 if s_R >= s_O - 0.01 else float(np.clip(s_R / (s_O + 1e-6), 0.0, 1.0))
    else:
        recovery = float(np.clip((s_R - s_H) / drop, 0.0, 1.0))

    # RevisitBridge: continuity at the H→R boundary
    emb_h_end   = _embed(model, preprocess, device, frames.get(h))
    emb_r_start = _embed(model, preprocess, device, frames.get(r))
    bridge = float(np.dot(emb_h_end, emb_r_start))

    # OcclusionDrop: H phase must actually move away from O phase baseline.
    # Low value → model never "left" target (H phase same as O) → Recovery invalid.
    occlusion_drop = float(np.clip((s_O - s_H) / (s_O + 1e-6), 0.0, 1.0))

    return {
        "revisit_recovery": round(recovery,       4),
        "revisit_bridge":   round(bridge,         4),
        "occlusion_drop":   round(occlusion_drop, 4),
        "s_O":              round(s_O, 4),
        "s_H":              round(s_H, 4),
        "s_R":              round(s_R, 4),
    }


@torch.no_grad()
def _embed_text(model, tokenizer, device, text: str) -> np.ndarray:
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().float().numpy()


def prompt_alignment_score(frames, prompt: str, n_sample: int = 5,
                           device: str = None) -> float:
    """
    CLIP text-image cosine similarity between the text prompt and sampled frames.
    Measures how well the generated video follows the input instruction.

    Returns mean cosine similarity over sampled frames (higher = better).
    """
    import open_clip

    model, preprocess, device = _load_clip(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text_emb = _embed_text(model, tokenizer, device, prompt)

    N = frames.num_frames
    indices = [int(i * (N - 1) / (n_sample - 1)) for i in range(n_sample)]

    sims = []
    for idx in indices:
        img_emb = _embed(model, preprocess, device, frames.get(idx))
        sims.append(float(np.dot(text_emb, img_emb)))

    return round(float(np.mean(sims)), 4)


def clip_frame_similarity(img_a: np.ndarray, img_b: np.ndarray,
                           device: str = None) -> float:
    """
    CLIP cosine similarity between two BGR frames.
    Used for GT-grounded R-phase comparison: generated frame vs GT frame.
    Returns a value in [0, 1].
    """
    model, preprocess, device = _load_clip(device)
    emb_a = _embed(model, preprocess, device, img_a)
    emb_b = _embed(model, preprocess, device, img_b)
    return float(np.clip(float(np.dot(emb_a, emb_b)), 0.0, 1.0))
