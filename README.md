# OPSD-V Project Page

Static project page for **OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators**.

## Local preview

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080`.

## Content

- `data.js`: authors, links, metrics, prompts, and curated cases.
- `assets/images`: paper figures and affiliation logos.
- `assets/videos`: synchronized Base/OPSD-V video pairs.
- `assets/posters`: 30-second poster frames for fast initial rendering.
- `scripts/prepare_assets.sh`: links selected source videos and renders posters on the remote machine.

Replace `project.paperUrl` and `project.codeUrl` in `data.js` when the public URLs are available.
