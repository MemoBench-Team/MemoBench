# Step 1: Automated Metrics — Supplementary Details

## Additional Usage Examples

**Use a specific GPU:**
```bash
python evaluation/run_eval.py \
    --mode synthetic \
    --gen_root output/my-model/Synthetic \
    --device cuda:1
```

**With prompt source (enables ImageReward scoring):**
```bash
python evaluation/run_eval.py \
    --mode synthetic \
    --gen_root output/my-model/Synthetic \
    --prompt_src_syn data/Synthetic_processed/metadata.csv
```

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `synthetic` | Data type: `synthetic`, `real`, or `both` |
| `--gen_root` | `output/Synthetic` | Root directory of generated frames (for single mode) |
| `--gen_root_syn` | — | Synthetic gen root (for `--mode both`) |
| `--gen_root_real` | — | Real gen root (for `--mode both`) |
| `--prompt_src_syn` | — | Synthetic prompt CSV (enables ImageReward) |
| `--prompt_src_real` | — | Real prompt CSV (enables ImageReward) |
| `--clip` | — | Evaluate only this clip ID (e.g., `Barnyard_001`) |
| `--scene` | — | Evaluate only this scene (e.g., `Barnyard`) |
| `--max_side` | `640` | Resize frames so the long side is at most this |
| `--sample_step` | `4` | Frame stride for temporal and quality metrics |
| `--device` | auto | CUDA device (`cuda`, `cuda:0`, `cuda:1`, `cpu`) |
| `--out_csv` | auto | Output path (default: `outputs/eval_{mode}_{timestamp}.csv`) |
| `--camera_ctrl` | `True` | Compute CameraControllability metric |

## Output

A CSV file saved to `outputs/eval_{mode}_{timestamp}.csv` with one row per clip and ~70 columns including all composite metrics (0–100), diagnostic metrics (raw scale), and per-phase pixel fidelity.

## What Happens Internally

1. Discovers all clip directories under `--gen_root`
2. Loads GT keyframes from `data/Synthetic_ExitReenter.xlsx` (or `Real_ExitReenter.xlsx`)
3. Maps GT phase boundaries (V/D/R) to the generated video's frame count
4. For each clip, computes all metrics on the appropriate phase frames:
   - **Visual Quality**: CLIP-IQA+ and LAION aesthetic on sampled frames (full video)
   - **Motion Smoothness**: RAFT optical flow warp error (V + R phases)
   - **Object Identity Consistency**: DINOv2 ViT-B/14 patch-token cosine similarity between frame 0 and R-phase frames
   - **Geo3D Consistency**: Depth Anything V2 monocular depth map cosine similarity (V + R phases)
   - **Camera Controllability**: Geodesic rotation error between MapAnything-estimated poses (from the generated video) and GT poses
   - **Pixel Fidelity**: PSNR, SSIM, LPIPS per phase (V, D, R) against GT frames
   - **ImageReward**: Human-preference score (requires `--prompt_src_*`)
