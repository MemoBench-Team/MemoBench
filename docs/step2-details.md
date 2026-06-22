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
4. Filters candidate masks by coverage (0.05%–50% of image area, falling back to 0.05%–70% if none qualify) to discard noise and background. The SAM-3 confidence threshold is set to 0.0, so masks are kept based on coverage rather than a score cutoff.
5. Marks the object as **detected** in a frame if any mask passes the coverage filter, using the highest-scoring mask's score as that frame's confidence. ORS is then computed as `detection_rate × mean_confidence`, where `detection_rate` is the fraction of R-phase frames with a detection and `mean_confidence` is the mean confidence over the detected frames. (No IoU or GT reference mask is involved — ORS is purely detection-based.)
