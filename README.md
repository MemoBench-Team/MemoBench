# MemoBench

**MemoBench** is a benchmark for evaluating *visual memory* in world generation models.
It tests whether models can faithfully regenerate objects after they leave and re-enter the camera's field of view, following a **V-D-R** (Visible → Disappeared → Reappear) paradigm.

The benchmark includes **360 clips** (196 synthetic + 164 real-world) spanning diverse scenes and physical state-change processes, evaluated across **14 metrics** covering visual quality, temporal consistency, geometric fidelity, object permanence, camera controllability, and VQA-based reasoning.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Output Directory Structure](#output-directory-structure)
- [Running Evaluation](#running-evaluation)
  - [Step 1: Automated Metrics](#step-1-automated-metrics)
  - [Step 2: Object Revisit Score (ORS)](#step-2-object-revisit-score-ors)
  - [Step 3: VQA Evaluation](#step-3-vqa-evaluation)
- [Leaderboard](#leaderboard)
- [Metrics Reference](docs/metrics-reference.md)
- [License](#license)

---

## Installation

### Requirements

- Linux (tested on Ubuntu 22.04)
- CUDA 12.x compatible GPU(s)
- Conda or Miniconda

### Environment Setup

```bash
conda create -n memobench python=3.11 -y
conda activate memobench

# PyTorch (adjust cu128 to match your CUDA driver version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Dependencies
pip install -r evaluation/requirements.txt

# MapAnything (required for Camera Controllability — estimates camera poses from generated videos)
git clone https://github.com/facebookresearch/map-anything.git third_party/map-anything
cd third_party/map-anything
pip install -e .
cd ../..

# SAM-3 (required for Object Revisit Score)
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
cd third_party/sam3
pip install -e .
cd ../..
```

### Verify Installation

```bash
conda activate memobench
python -c "import torch; import open_clip; import lpips; print('OK')"
```

---

## Data Preparation

Download the MemoBench dataset and place it under the `data/` directory:

```
MemoBench/
├── data/
│   ├── Synthetic_processed/                # Synthetic GT data (196 clips)
│   │   ├── Barnyard/
│   │   │   ├── Barnyard_001/
│   │   │   │   ├── image.jpg               # Reference start frame
│   │   │   │   ├── intrinsics.npy          # (N_gt, 4) [fx, fy, cx, cy]
│   │   │   │   └── poses.npy              # (N_gt, 4, 4) cam extrinsics (c2w)
│   │   │   ├── Barnyard_002/
│   │   │   └── ...
│   │   ├── CityPark/
│   │   ├── CityStreet/
│   │   ├── Japanese_Street/
│   │   ├── Miami/
│   │   ├── Nordic/
│   │   ├── NYstreet/
│   │   ├── OldFactory/
│   │   ├── WinterTown/
│   │   ├── ZenGarden/
│   │   ├── Synthetic_ExitReenter.xlsx      # V-D-R phase boundaries
│   │   └── Real_ExitReenter.xlsx           # V-D-R phase boundaries
│   │
│   ├── Real_Raw/                           # Real-world GT data (164 clips)
│   │   ├── 001/
│   │   │   ├── 001.mp4                     # GT video
│   │   │   ├── 001-intrinsics.json         # {"camera": {fx, fy, cx, cy, ...}}
│   │   │   └── timestamps.txt             # One timestamp per GT frame
│   │   ├── 002/
│   │   └── ...
│   ├── mapanything/                        # Pre-estimated GT camera poses
│   │   └── outputs/
│   │       └── real/                       # {clip_id}_droid.npz per real clip
│   │
│   ├── sam3_metadata/                      # Object metadata for ORS
│   │   ├── synthetic_metadata.csv          # scene, video_id, subject, prompt
│   │   └── real_metadata.csv              # video_id, subject, prompt
│   │
│   └── vqa_questions/                      # Pre-generated & filtered VQA
│       ├── Barnyard-1-questions.csv        # Per-clip question sets
│       ├── Barnyard-2-questions.csv
│       └── ...                             # 44 files total
```

**Key files explained:**
- `Synthetic_ExitReenter.xlsx` / `Real_ExitReenter.xlsx`: Define the V-D-R phase boundaries per clip. Columns include `exits` (frame where object leaves FOV) and `re-enter` (frame where object returns). These are in GT frame space and are automatically scaled to match the generated video's frame count.
- `poses.npy` (synthetic) / `mapanything/outputs/real/` (real): GT camera poses. Synthetic poses come from the UE5 renderer. Real-world GT poses are pre-estimated using MapAnything and provided with the dataset — you do not need to run MapAnything on GT videos yourself.
- `sam3_metadata/*.csv`: Contains the text description of the target object per clip (e.g., "fox", "person in silver suit"), used as the SAM-3 text prompt for ORS.
- `vqa_questions/*.csv`: Pre-generated and filtered Yes/No question banks (6 per dimension × 4 dimensions).

---

## Output Directory Structure

Place your model's generated video frames in the following structure:

```
output/{model_name}/
├── Synthetic/
│   ├── {Scene}/
│   │   ├── {Scene}_{NNN}/
│   │   │   └── frames/
│   │   │       ├── 00000.png
│   │   │       ├── 00001.png
│   │   │       └── ...
│   │   └── ...
│   └── ...
│
└── Real/
    ├── {NNN}/
    │   └── frames/
    │       ├── 00001.png
    │       ├── 00002.png
    │       └── ...
    └── ...
```

**Requirements:**
- **Format:** PNG images, one frame per file.
- **Naming:** Zero-padded integers (e.g., `00000.png`). The evaluation pipeline sorts frames lexicographically.
- **Synthetic scenes:** Barnyard, CityPark, CityStreet, Japanese_Street, Miami, Nordic, NYstreet, OldFactory, WinterTown, ZenGarden (and others). Clip IDs follow `{Scene}_{NNN}` (e.g., `Barnyard_001`).
- **Real clips:** Numbered directories (e.g., `001/`, `009/`, `120/`).

---

## Running Evaluation

The full evaluation pipeline has three steps:

```
Step 1: run_eval.py      → automated metrics CSV
Step 2: compute_ors.py   → ORS scores CSV
Step 3: llm-vqa.py       → VQA scores CSVs
        leaderboard.py   → unified leaderboard
```

All commands below assume you are in the `MemoBench/` root directory with the `memobench` environment activated.

```bash
conda activate memobench
```

### Step 1: Automated Metrics

Computes 13 metrics per clip: VisualQuality, MotionSmoothness, ObjIdentityConsistency, Geo3DConsistency, CameraControllability, ImageRewardScore, and per-phase pixel fidelity (PSNR, SSIM, LPIPS for V/D/R phases).

**Evaluate synthetic clips:**
```bash
python evaluation/run_eval.py \
    --mode synthetic \
    --gen_root output/my-model/Synthetic
```

**Evaluate real-world clips:**
```bash
python evaluation/run_eval.py \
    --mode real \
    --gen_root output/my-model/Real
```

**Evaluate both at once:**
```bash
python evaluation/run_eval.py \
    --mode both \
    --gen_root_syn output/my-model/Synthetic \
    --gen_root_real output/my-model/Real
```

**Evaluate a single clip (useful for debugging):**
```bash
python evaluation/run_eval.py \
    --mode synthetic \
    --gen_root output/my-model/Synthetic \
    --clip Barnyard_001
```

For all command-line options, output format, and implementation details, see [Step 1 Supplementary Details](docs/step1-details.md).

### Step 2: Object Revisit Score (ORS)

ORS measures whether the model regenerates the target object when the camera returns in the R-phase. It uses SAM-3 text-prompted segmentation with the object description from `data/sam3_metadata/`.

**Before running**, register your model in `compute_ors.py` by adding an entry to the `MODEL_CONFIGS` dict:
```python
MODEL_CONFIGS = {
    # ...existing models...
    "MyModel": {
        "synthetic": ["output/my-model/Synthetic"],
        "real":      ["output/my-model/Real"],
    },
}
```

**Run ORS for your model:**
```bash
python evaluation/compute_ors.py --model-name MyModel
```

**Run ORS for all registered models:**
```bash
python evaluation/compute_ors.py --model-name all
```

**Custom output directory:**
```bash
python evaluation/compute_ors.py \
    --model-name MyModel \
    --output-dir results/ors/
```

**Resume from partial run (skip already-scored clips):**
```bash
python evaluation/compute_ors.py \
    --model-name MyModel \
    --skip-existing
```

**SAM-3 checkpoint location:** By default, `compute_ors.py` looks for SAM-3 at `third_party/sam3/`. To override, set the `SAM3_DIR` environment variable:
```bash
SAM3_DIR=/path/to/sam3 python evaluation/compute_ors.py --model-name MyModel
```

For all command-line options, output format, and implementation details, see [Step 2 Supplementary Details](docs/step2-details.md).

### Step 3: VQA Evaluation

The VQA pipeline uses a Vision-Language Model (VLM) to answer pre-generated Yes/No questions about each generated video. It evaluates four dimensions: Instruction Following, Object & Background, Continuity of Memory, and Physics Adherence.

**Prerequisites:**
- A Gemini API key (set as environment variable `GEMINI_API_KEY`)
- Pre-generated question banks in `data/vqa_questions/` (provided with the dataset; to regenerate, see `evaluation/vqa/llm-judger.py`)

**Before running**, register your model's video directories in `evaluation/vqa/llm-vqa.py` by adding to `MODEL_CONFIGS`:
```python
MODEL_CONFIGS = {
    # ...existing models...
    "MyModel": [
        "output/my-model/Synthetic",
        "output/my-model/Real",
    ],
}
```

**Run VQA evaluation:**
```bash
export GEMINI_API_KEY="your-api-key-here"

python evaluation/vqa/llm-vqa.py \
    --model-name MyModel
```

**Resume from partial run:**
```bash
python evaluation/vqa/llm-vqa.py \
    --model-name MyModel \
    --skip-existing
```

**Run all registered models:**
```bash
bash evaluation/vqa/run_all_vqa.sh
```

For all command-line options, output format, and implementation details, see [Step 3 Supplementary Details](docs/step3-details.md).

---

## Leaderboard

After completing all three evaluation steps, aggregate results into a unified leaderboard:

```bash
python leaderboard/leaderboard.py \
    --eval-dir  outputs/ \
    --ors-dir   ors_results/ \
    --vqa-dir   vqa_results/ \
    --output    leaderboard.csv
```

The leaderboard script automatically:
- Loads the latest eval CSV per model from `--eval-dir/{model}/eval_*.csv`
- Loads ORS scores from `--ors-dir/{model}/ors_scores.csv`
- Parses per-clip VQA scores from `--vqa-dir/{model}/*.csv`
- Merges by model name (handles different naming conventions automatically)
- Converts all scores to their display scale (ORS: 0–1 → 0–100; VQA: 0–1 → 0–100)

**Display and export:**
```bash
# View in terminal (default), or export to other formats
python leaderboard/leaderboard.py --load leaderboard.csv
python leaderboard/leaderboard.py --load leaderboard.csv --format html    # sortable HTML
python leaderboard/leaderboard.py --load leaderboard.csv --format markdown
python leaderboard/leaderboard.py --load leaderboard.csv --format latex
python leaderboard/leaderboard.py --load leaderboard.csv --sort ORS      # sort by any metric
```

---

## Metrics Reference

For detailed descriptions of all 14 metrics (backbones, phases, score ranges), see the [Metrics Reference](docs/metrics-reference.md).

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

