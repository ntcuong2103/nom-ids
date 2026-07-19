# nom-ids

Training utilities for Nom character IDS recognition.

## Environment

This project is packaged for `uv` and targets the existing `cloudspace` conda
environment baseline:

- Python 3.10
- PyTorch 2.7
- torchvision 0.22
- PyTorch Lightning 2.5

Install the package into the existing conda environment with:

```bash
conda activate cloudspace
UV_CACHE_DIR=/tmp/uv-cache uv pip install -e .
```

Or create a local `uv` virtual environment with:

```bash
uv sync
```

If the Cloudspace home cache is read-only, use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

Run the existing training entry point with:

```bash
uv run nom-ids-train
```

Copy the NomNaOCR page images into the expected image directory with:

```bash
uv run nom-ids-extract-data
```
