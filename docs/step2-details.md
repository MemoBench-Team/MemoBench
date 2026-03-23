# Step 2: Object Revisit Score (ORS) — Supplementary Details

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | required | Model name (must match `MODEL_CONFIGS` key), or `all` |
| `--output-dir` | `ors_results` | Output directory |
| `--skip-existing` | off | Skip clips that already have scores |

## Output

`ors_results/{ModelName}/ors_scores.csv` with columns:

| Column | Description |
|--------|-------------|
| `clip_id` | Clip identifier (e.g., `Barnyard_001` or `009`) |
| `data_type` | `synthetic` or `real` |
| `scene` | Scene name |
| `n_gen` | Number of generated frames |
| `r_start` | R-phase start frame index |
| `ors` | Object Revisit Score (0–1) |
| `detection_rate` | Fraction of R-phase frames where object was detected |
| `mean_confidence` | Mean SAM-3 detection confidence |
| `n_r_frames` | Number of R-phase frames evaluated |
| `n_detected` | Number of frames with valid detection |

## What Happens Internally

1. Loads object descriptions from `data/sam3_metadata/synthetic_metadata.csv` and `real_metadata.csv`
2. For each clip, identifies R-phase frames using `Synthetic_ExitReenter.xlsx` / `Real_ExitReenter.xlsx`
3. Runs SAM-3 text-prompted segmentation on each R-phase frame with the object description as prompt
4. Filters masks by coverage (0.05%–50% of image area) to discard noise and background
5. Computes ORS as the maximum IoU between detected masks and a resized GT reference mask
