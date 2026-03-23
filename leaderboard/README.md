# MemoBench Leaderboard

Aggregate per-model evaluation results into a ranked leaderboard table.

## Metrics

| Category | Metric | Range | Description |
|----------|--------|-------|-------------|
| **Automated** | VisQual | 0–100 | Visual quality (CLIP-IQA+ + LAION aesthetic) |
| | MotSmooth | 0–100 | Motion smoothness (RAFT optical flow) |
| | ObjConsist | 0–100 | Object identity consistency (DINOv2) |
| | 3DConsist | 0–100 | Geometric 3D consistency (Depth Anything V2) |
| | CamCtrl | 0–100 | Camera controllability (ATE vs GT poses) |
| | ImgReward | 0–100 | Image reward (human preference) |
| **Object Memory** | ORS | 0–100 | Object Revisit Score (SAM-3 detection) |
| **Pixel Fidelity** | PSNR | dB | Peak signal-to-noise ratio |
| | SSIM | 0–1 | Structural similarity |
| | LPIPS ↓ | 0–1 | Perceptual distance (lower is better) |
| **VQA** | VQA-IF | 0–100 | Instruction following |
| | VQA-OB | 0–100 | Object & background consistency |
| | VQA-CM | 0–100 | Continuity of memory |
| | VQA-PA | 0–100 | Physics adherence |
| | VQA-All | 0–100 | Overall VQA score |

## Usage

### Build from evaluation outputs

```bash
python leaderboard.py \
    --eval-dir path/to/outputs/ \
    --ors-dir  path/to/ors_results/ \
    --vqa-dir  path/to/vqa_results/ \
    --output   leaderboard.csv
```

### Display formats

```bash
# Terminal (colored, default)
python leaderboard.py --load leaderboard.csv

# Sorted by a specific metric
python leaderboard.py --load leaderboard.csv --sort ORS

# GitHub Markdown table
python leaderboard.py --load leaderboard.csv --format markdown

# LaTeX table (for paper)
python leaderboard.py --load leaderboard.csv --format latex

# CSV
python leaderboard.py --load leaderboard.csv --format csv
```

### Directory structure

```
outputs/
├── lingbot-world/
│   └── eval_*.csv
├── wan2.2/
│   └── eval_*.csv
└── ...

ors_results/
├── LingBot-World/
│   └── ors_scores.csv
└── ...

vqa_results/
├── lingbot-world/
│   └── vqa_scores.csv
└── ...
```

## Adding a new model

1. Run `evaluation/run_eval.py` to produce the eval CSV
2. Run `evaluation/compute_ors.py` to produce ORS scores
3. Run the VQA pipeline to produce VQA scores
4. Re-run `leaderboard.py` — the new model is picked up automatically
