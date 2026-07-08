# MemoBench

**MemoBench** is a benchmark for evaluating *visual memory* in world generation models.
It tests whether models can faithfully regenerate objects after they leave and re-enter the camera's field of view, following a **V-D-R** (Visible → Disappeared → Reappear) paradigm.

The benchmark includes **360 clips** (196 synthetic + 164 real-world) spanning diverse scenes and physical state-change processes, evaluated across **14 metrics** covering visual quality, temporal consistency, geometric fidelity, object permanence, camera controllability, and VQA-based reasoning.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Inference Protocol](#inference-protocol)
- [Output Directory Structure](#output-directory-structure)
- [Running Evaluation](#running-evaluation)
  - [Step 1: Automated Metrics](#step-1-automated-metrics)
  - [Step 2: Object Revisit Score (ORS)](#step-2-object-revisit-score-ors)
  - [Step 3: VQA Evaluation](#step-3-vqa-evaluation)
- [Leaderboard](#leaderboard)
- [Metrics Reference](docs/metrics-reference.md)
- [License](#license)
- [Citation](#citation)

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
│   │   │   ├── Barnyard_001/               # (digit suffix below = clip number, e.g. RGB5/ for _005)
│   │   │   │   ├── RGB1/                   # GT RGB frames; first frame = reference start frame
│   │   │   │   ├── Depth1/                 # GT metric depth maps
│   │   │   │   ├── 1/
│   │   │   │   │   ├── intrinsics.npy      # (N_gt, 4) [fx, fy, cx, cy]
│   │   │   │   │   └── poses.npy           # (N_gt, 4, 4) cam extrinsics (c2w)
│   │   │   │   └── camera_full1.csv        # raw UE5 camera track
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
│   │   └── Synthetic_ExitReenter.xlsx      # V-D-R phase boundaries (synthetic)
│   │
│   ├── Real_Raw/                           # Real-world GT data (164 clips)
│   │   ├── 001/
│   │   │   ├── 001.mp4                     # GT video (also defines the GT frame count)
│   │   │   ├── FRAMES/                     # extracted GT frames (frame_0001.jpg, ...)
│   │   │   ├── depth/                      # per-frame depth maps
│   │   │   ├── raw/
│   │   │   │   └── 001_mapanything.npz     # MapAnything GT camera: cam_c2w + intrinsic
│   │   │   └── thresholded/
│   │   │       └── 001_mapanything.npz     # alternative estimate (not used by eval)
│   │   ├── 002/
│   │   ├── ...
│   │   └── Real_ExitReenter.xlsx           # V-D-R phase boundaries (real)
│   │
│   ├── sam3_metadata/                      # Per-clip prompts, first-frame paths, ORS subjects
│   │   ├── synthetic_metadata.csv          # scene, video_id, subject, exr_path (first-frame path), prompt
│   │   └── real_metadata.csv               # video_id, subject, full_image_path (first-frame path), prompt
│   │
│   └── vqa_questions/                      # Pre-generated & filtered VQA
│       ├── Barnyard-1-questions.csv        # Per-clip question sets (45 files)
│       ├── Barnyard-2-questions.csv
│       ├── ...
│       └── failure-cases.csv               # VQA clip list: scene, video_id, folder, hint, prompt
```

**Key files explained:**
- `Synthetic_ExitReenter.xlsx` (under `Synthetic_processed/`) / `Real_ExitReenter.xlsx` (under `Real_Raw/`): Define the V-D-R phase boundaries per clip — the frame where the object leaves the FOV and the frame where it returns. The two files use different column names: the synthetic sheet uses `exits` / `re-enter`, the real sheet uses `exit` / `reenter`. These are in GT frame space and are automatically scaled to match the generated video's frame count. **These sheets are used at evaluation time only** — they are not model inputs. Note that `Real_ExitReenter.xlsx` has no `prompt` column: the per-clip generation inputs for the Real split (text prompt + reference first frame) live in `data/sam3_metadata/real_metadata.csv` (see [Inference Protocol](#inference-protocol)).
- `{n}/poses.npy` (synthetic) / `Real_Raw/{id}/raw/{id}_mapanything.npz` (real): GT camera poses. Synthetic poses come from the UE5 renderer (with the raw camera track in `camera_full{n}.csv`). Real-world GT poses and intrinsics are pre-estimated using MapAnything and shipped per clip in the `raw/` subfolder — you do not need to run MapAnything on GT videos yourself.
- `sam3_metadata/*.csv`: Per-clip metadata serving two roles. (1) **Generation inputs**: the `prompt` column holds the full text prompt for each clip, and the reference first frame is given by `exr_path` (synthetic — historical column name, the path points to the first `RGB{n}/` JPG) / `full_image_path` (real) — see [Inference Protocol](#inference-protocol). (2) **Evaluation**: the `subject` column (e.g., "fox", "person in silver suit") is used as the SAM-3 text prompt for ORS.
- `vqa_questions/*-questions.csv`: Pre-generated and filtered Yes/No question banks (6 per dimension × 4 dimensions).
- `vqa_questions/failure-cases.csv`: The clip list the VQA step iterates (columns: `scene`, `video_id`, `hint`, `folder`, `Finished`, `prompt`). `folder` is the generated-frames directory name for each clip; `llm-vqa.py` reads this file by default via `--cases-csv`.

---

## Inference Protocol

How to generate videos for MemoBench with your own model. The rule: **every model is conditioned on the same first frame + text prompt; models that support camera conditioning additionally receive the GT trajectory.** Keeping these inputs identical across models is what makes results comparable.

**Per-clip generation inputs:**

| Split | Text prompt | Reference first frame | Camera (optional) |
|-------|-------------|----------------------|-------------------|
| Synthetic | `prompt` in `data/sam3_metadata/synthetic_metadata.csv` (same text as `Synthetic_ExitReenter.xlsx`) | `exr_path` in the same CSV (points to the first `RGB{n}/` JPG) | `intrinsics.npy` + `poses.npy` (`(N,4,4)` c2w) per clip |
| Real | `prompt` in `data/sam3_metadata/real_metadata.csv` | `full_image_path` in the same CSV | `Real_Raw/{id}/raw/{id}_mapanything.npz` (`cam_c2w` + `intrinsic`) |

**By model interface:**

- **CI2V (camera-controllable image-to-video)**: first frame + prompt + GT camera trajectory.
- **Explicit-3D view synthesis**: first frame + explicit GT camera poses.
- **I2V / TI2V (no camera support)**: first frame + prompt only. Do not feed camera information. Such models are still scored on Camera Controllability and Instruction Following (and will score low there) — that is expected and reported as-is.

Notes:

- The V-D-R phase boundaries (`*_ExitReenter.xlsx`) are **evaluation-time only** and are automatically rescaled to your generated video's frame count — you do not need to align frames to them during generation.
- The GT camera poses serve double duty: a generation input for camera-conditioned models, and the reference for the Camera Controllability metric (poses are re-estimated from your generated video with MapAnything and compared against GT) for all models.

After generation, arrange your frames as described in [Output Directory Structure](#output-directory-structure) and run the three-step evaluation below.

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

Computes the automated metrics per clip: VisualQuality, MotionSmoothness, ObjIdentityConsistency, Geo3DConsistency, CameraControllability, ImageRewardScore, and pixel fidelity (PSNR, SSIM, LPIPS) — reported overall and per V/D/R phase.

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

**Join the leaderboard:**

:trophy: Check out our [Leaderboard on Huggingface](https://huggingface.co/spaces/cavenqjw/MemoBench_Leaderboard) for the latest rankings and numerical results!

How to join the leaderboard:
- **Submit Results (Highly Recommended):** Follow the evaluation instructions above to generate your `leaderboard.csv` file. Email it to tonychen54816@gmail.com, and the MemoBench Team will update the leaderboard.
- **Submit Video Samples:** Alternatively, you can send us your raw video samples for us to evaluate. Please note that processing times for this method will vary depending on our team's bandwidth and resources.

---

## Metrics Reference

For detailed descriptions of all 14 metrics (backbones, phases, score ranges), see the [Metrics Reference](docs/metrics-reference.md).

---

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

---

## Citation

If you find MemoBench useful in your research, please cite:

```bibtex
@article{chen2026memobench,
  title={MemoBench: Benchmarking World Modeling in Dynamically Changing Environments},
  author={Chen, Haoyu and Zhou, Kaichen and Hua, Hang and Zhang, Kaile and Qian, Jingwen and Ma, Wufei and Chen, Haonan and Liu, Chunjiang and Zhao, Yizhou and Wang, Xiaoyuan and others},
  journal={arXiv preprint arXiv:2606.27537},
  year={2026}
}
```

