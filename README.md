# Junction Reconstruction

Reconstruct lane structure from a junction image and write OpenDRIVE. Run from the repository root (Python 3.10+).

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=<your-key>
```

Set the model id in `MODELS` in `scripts/methods/llm_kg/main.py`. Add that same id under `openrouter` in `scripts/methods/llm_pricing.py` with USD per 1M tokens (`input`, `output`, `cache_read`, `cache_write`). A model with no price entry still runs; its cost is null.

Calls go through the local proxy `http://127.0.0.1:2022`. For a direct connection, set `DEFAULT_PROXY = None` in `scripts/methods/api_providers.py`.

## Run

```bash
python -m scripts.methods.llm_kg.main
```

`--parallel` overrides the default of 4 concurrent cases. `DATASETS` is `openaerialmap`; only case folders that contain `arms.json` are included. Junction detection is skipped, so each image is one junction. The run then does arm level, lane level, refinement, OpenDRIVE export, and evaluation. CARLA and esmini are off.

A case is `dataset/<dataset>/<case_id>/`. The image and `meta.json` (`driving_side`) are inputs. `arms.json` and `lanes.json` are ground truth for scoring.

Results are written to `scripts/methods/llm_kg/results/<dataset>/<case_id>/`. The summary is `scripts/methods/llm_kg/results/<dataset>/case_metrics.csv`.

Re-score existing outputs with `python -m scripts.methods.llm_kg.eval`.

## Optional

`scripts/data_process/` needs the optional packages in `requirements.txt`. CARLA validation needs the CARLA Python API from a CARLA install.

For an esmini preview, place the `esmini` folder next to `scripts` and set `RUN_ESMINI = True` in `scripts/methods/llm_kg/main.py`. The binary is `esmini/bin/esmini`.
