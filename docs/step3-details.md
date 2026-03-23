# Step 3: VQA Evaluation — Supplementary Details

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | required | Model name (must match `MODEL_CONFIGS` key) |
| `--video-dirs` | from config | Override video directories (space-separated) |
| `--output-dir` | `vqa_results/` | Root output directory |
| `--questions-dir` | `data/vqa_questions/` | Directory with per-clip question CSVs |
| `--cases-csv` | `failure-cases.csv` | Path to failure cases CSV |
| `--skip-existing` | off | Skip clips that already have scores |
| `--model-id` | `gemini-3-flash-preview` | VLM model ID |

## Output

Per-clip CSV files in `vqa_results/{ModelName}/`:

```
vqa_results/MyModel/
├── Barnyard-1.csv
├── Barnyard-2.csv
├── CityPark-6.csv
└── ...
```

Each CSV contains columns: `scene`, `video_id`, `Evaluation` (full JSON with per-question verdicts), and `score` (JSON dict with per-dimension scores 0–1):

```json
{
  "Instruction Following": 0.667,
  "Object and Background": 0.333,
  "Continuity of Memory": 0.600,
  "Physics Adherence": 0.667
}
```
