# Data Processing

This folder contains a minimal, audio-free preprocessing pipeline for building
the LMDB format used by OPSD-V training. It is provided so users can prepare
their own long-video data. No videos, prompts, latents, embeddings, or LMDB
files are included in this repository.

The pipeline has four steps:

1. Collect your own long videos.
2. Convert videos to normalized `.pt` tensors and pair each video with a prompt.
3. Encode video tensors into Wan VAE latents.
4. Encode text prompts and pack prompts, prompt embeddings, and VAE latents into
   an LMDB.

## Expected Directory Layout

After preprocessing, a dataset root should look like this:

```text
dataset_root/
├── manifest.jsonl
├── raw_videos/
│   ├── sample_000.mp4
│   └── sample_001.mp4
├── processed_video/
│   ├── 00000000_video.pt
│   └── 00000001_video.pt
├── prompts/
│   ├── 00000000_prompt.txt
│   └── 00000001_prompt.txt
├── processed_latent/
│   ├── 00000000_latent.pt
│   └── 00000001_latent.pt
└── lmdb_prompt/
    ├── data.mdb
    └── lock.mdb
```

For the default 480 x 832 setting with 243 latent frames, the final LMDB stores:

```text
latents_shape         N 243 16 60 104
prompt_embeds_shape   N 512 4096
prompts_shape         N
```

## Step 1: Collect Long Videos

Create a JSONL manifest with one sample per line:

```jsonl
{"id": "00000000", "video": "raw_videos/sample_000.mp4", "prompt": "A detailed prompt describing the long video."}
{"id": "00000001", "video": "raw_videos/sample_001.mp4", "prompt": "Another detailed long-video prompt."}
```

`id` is optional. If omitted, the script assigns zero-padded ids in manifest
order. `video` can be absolute or relative to the manifest file.

## Step 2: Convert Videos to `.pt` and Prompts

```bash
python data_processing/prepare_videos_and_prompts.py \
  --manifest /path/to/dataset_root/manifest.jsonl \
  --output_root /path/to/dataset_root \
  --num_frames 243 \
  --height 480 \
  --width 832 \
  --fps 16
```

This writes:

```text
processed_video/{sample_id}_video.pt
prompts/{sample_id}_prompt.txt
```

The saved video tensor has shape `[T, C, H, W]`, dtype `float16`, and values in
`[-1, 1]`. If a source video has more than `num_frames`, frames are sampled
uniformly. If it has fewer frames, the last frame is repeated.

## Step 3: Encode Wan VAE Latents

Download Wan2.1-T2V-1.3B first:

```bash
mkdir -p checkpoints
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir checkpoints/Wan2.1-T2V-1.3B
```

Then run:

```bash
WAN_MODEL_ROOT=/path/to/Wan2.1-T2V-1.3B \
CUDA_VISIBLE_DEVICES=0 \
python data_processing/compute_vae_latents.py \
  --video_pt_dir /path/to/dataset_root/processed_video \
  --output_latent_folder /path/to/dataset_root/processed_latent \
  --resume
```

For multiple GPUs:

```bash
WAN_MODEL_ROOT=/path/to/Wan2.1-T2V-1.3B \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  data_processing/compute_vae_latents.py \
  --video_pt_dir /path/to/dataset_root/processed_video \
  --output_latent_folder /path/to/dataset_root/processed_latent \
  --resume
```

This writes:

```text
processed_latent/{sample_id}_latent.pt
```

Each latent is saved as `[1, T, 16, H_lat, W_lat]`.

## Step 4: Encode Text and Create LMDB

```bash
WAN_MODEL_ROOT=/path/to/Wan2.1-T2V-1.3B \
CUDA_VISIBLE_DEVICES=0 \
python data_processing/create_lmdb.py \
  --latent_path /path/to/dataset_root/processed_latent \
  --prompt_dir /path/to/dataset_root/prompts \
  --lmdb_path /path/to/dataset_root/lmdb_prompt \
  --encode_prompt_embeds \
  --prompt_embeds_fp16
```

The LMDB will contain:

```text
latents_{i}_data
prompts_{i}_data
prompt_embeds_{i}_data
latents_shape
prompts_shape
prompt_embeds_shape
```

This is the format consumed by the OPSD-V training configs through
`data_path: /path/to/dataset_root/lmdb_prompt`.

## Notes

- This release does not include company training data.
- Audio embeddings are intentionally not part of this open-source preprocessing
  path.
- If you use different resolution or frame count, update training configs
  accordingly.
- Prompt quality matters. In our experiments, prompts are long-form captions
  that describe the full video rather than only the first frame.
