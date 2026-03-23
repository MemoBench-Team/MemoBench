# Metrics Reference

## Automated Metrics (0–100 scale)

| Metric | Short | Description | Backbone | Phases |
|--------|-------|-------------|----------|--------|
| Visual Quality | VisQual | Perceptual quality (CLIP-IQA+ + LAION aesthetic) | CLIP ViT-B-32 | Full video |
| Motion Smoothness | MotSmooth | Temporal coherence via optical flow warp error | RAFT | V + R |
| Object Identity Consistency | ObjConsist | Whether the object looks the same after reappearing | DINOv2 ViT-B/14 | R vs frame 0 |
| Geo3D Consistency | 3DConsist | Geometric depth map consistency | Depth Anything V2 | V + R |
| Camera Controllability | CamCtrl | How well the generated video follows the input camera trajectory | MapAnything | Full video |
| Image Reward | ImgReward | Human-preference alignment score | ImageReward | Full video |

## Object Revisit Score (0–100 scale)

| Metric | Short | Description | Backbone | Phases |
|--------|-------|-------------|----------|--------|
| Object Revisit Score | ORS | Whether the target object is detected when the camera returns | SAM-3 | R only |

## Pixel Fidelity (native units)

| Metric | Description | Range | Phases |
|--------|-------------|-------|--------|
| PSNR | Peak signal-to-noise ratio vs GT | dB (higher = better) | Per-phase (V/H/R) + All |
| SSIM | Structural similarity vs GT | 0–1 (higher = better) | Per-phase (V/H/R) + All |
| LPIPS | Perceptual distance vs GT | 0–1 (lower = better) | Per-phase (V/H/R) + All |

## VQA Metrics (0–100 scale)

| Metric | Short | Description |
|--------|-------|-------------|
| Instruction Following | VQA-IF | Did the video follow the requested camera movements and events? |
| Object & Background | VQA-OB | Are object identity and background details consistent? |
| Continuity of Memory | VQA-CM | Does the model preserve object state while out of frame? |
| Physics Adherence | VQA-PA | Are lighting, shadows, and motion physically plausible? |
