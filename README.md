# AgentSeg

(formerly AutoSeg) An annotation tool for image segmentation with SAM integration and automatic propagation.

## Features

- Manual bounding box annotation
- SAM-powered automatic segmentation from bounding boxes
- Encoder-based annotation propagation across sequential images
- YOLO-seg format export
- Review workflow for propagated annotations

## Demo

https://github.com/user-attachments/assets/946ba046-a91b-4d39-ab81-d85eb0bfba0c

## Installation

### Prerequisites

- **Python**: 3.12+
- **uv**: recommended
- **CUDA**: 12+
- **Node.js/Bun**: For frontend

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone --recurse-submodules https://github.com/talhabw/autoseg.git
   cd autoseg
   ```

2. **Install Python dependencies**:
   ```bash
   uv sync
   ```
   *Note: This creates a `.venv` with Python 3.12 and installs all required packages.*

3. **Install Frontend**:
   ```bash
   cd frontend
   bun install
   ```

4. **Download Models**:
   Place your model weights in the `models/` directory in the root. The application expects specific filenames:
   - **SAM3**: Downloaded automatically on first use. [(may require hf cli authentication)](https://huggingface.co/facebook/sam3)
   - **DINOv3**: [Download from Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
     - `dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth` (ViT-H/16)
     - `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L/16)
     - `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (ViT-B/16)
   - **Pixio**: [Download from HuggingFace](https://huggingface.co/collections/facebook/pixio) 
     - `pixio_vitb16.pth`
     - `pixio_vitl16.pth`
     - `pixio_vith16.pth`
     - `pixio_vit1b16.pth`

## Usage

Start everything with one command:
```bash
uv run agentseg up        # or .venv/bin/agentseg up
```
This starts whatever is not running yet and prints the web UI URL
(http://localhost:5172 by default):
- **sam_server** on :8770 (SAM 3 video + image heads, needs `models/sam3/sam3.pt`; first load takes 1-2 min),
- the **web UI**, rebuilt with bun when `frontend/` changed since the last build,
- the **backend** on :5172, which serves the API and the built web UI.

Logs go to `.agentseg/` (`sam_server.log`, `backend.log`, `frontend_build.log`).
Useful flags: `--no-sam`, `--no-ui`, `--rebuild`, `--host 0.0.0.0` (expose on the network).
`agentseg status` shows what is up; `agentseg down` stops the backend and
`agentseg down --all` also stops sam_server (frees the GPU).

The same `agentseg` CLI drives labeling from the terminal or an agent
(see `.agent/skills/agentseg-label/SKILL.md`; `.agents/skills/agentseg-label`
links to it so Muse Code finds it).

**Labeling agent (Muse Code).** The backend runs `muse exec --json` headless in
the repo root (model `muse-spark-1.3-contributor`) and streams what the agent does into the
web UI. Muse reads its API key from `META_API_KEY`:
```bash
read -rs META_API_KEY && export META_API_KEY   # paste the key, Enter; not echoed
agentseg down && agentseg up                   # the backend must see the key
agentseg agent run "label the torpedo" --images /path/to/frames
```
Instead of exporting, you can put `META_API_KEY=...` in `.env` (gitignored, see
`.env.example`); the backend reads it when a run starts. `agentseg agent status`
and `agentseg agent stop` manage runs. Settings: `AGENTSEG_AGENT_MODEL`,
`AGENTSEG_AGENT_EFFORT`, `AGENTSEG_MUSE_BIN`, `AGENTSEG_AGENT_TIMEOUT`.
Transcripts go to `.agentseg/agent_runs/<id>/`; open `http://localhost:5172/?replay=<id>`
to play a run back in the UI.

Example: [docs/media/muse-second-try.mp4](docs/media/muse-second-try.mp4) shows a
replayed run. The first box also takes the stand's legs, so the agent marks
the legs as background and the mask is fixed. The run's steps are scripted, but the
masks come from SAM.

**Frontend development** (hot reload, proxies `/api` to the backend on :5172):
```bash
cd frontend
bun run dev               # http://localhost:5173
```

## Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev python -m pytest
cd frontend && bun run test
```

## Acknowledgements

This project builds upon the following excellent third-party projects and model releases:

- **SAM3**: Segment Anything model used for SAM-powered mask generation.
- **DINOv3 / Pixio**: Vision encoder used for feature extraction / propagation.

I am grateful to the authors and maintainers of these projects for releasing their work.

## License

This project depends on third-party components with their own licenses:

- **Pixio**: [Facebook license](https://github.com/facebookresearch/pixio/blob/main/LICENSE).
- **DINOv3**: [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
- **SAM3**: [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

Please review all licenses before use.

Your rights to use, redistribute, or deploy this project “end-to-end” may be constrained by the licenses of the third-party model code and/or weights you choose to use. Please review and comply with the applicable third-party licenses before use, especially for redistribution or commercial deployment.

Where this repository does not directly redistribute third-party weights, users are expected to obtain them separately and agree to the corresponding license terms from the original providers.
