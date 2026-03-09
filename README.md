# Usage guide

## Install uv
Install uv, see [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## cd into repository
```
cd ./mobilenet/
```

## Install dependencies
```
uv sync
```

## Activate virtual environment
```
source .venv/bin/activate
```

## Download images from Scryfall and build embedded sqlite DB (mobilecache) based on MTGJSON db
```
uv run python -m mtg_recognition.jobs.build_database
```

## Build embeddings and perceptual hashing indexes
```
uv run python -m mtg_recognition.engine.pipeline
```
**Note**: On first run, this command will also build indexes (embeddings and phashes).

## Run main app

```
uv run main.py
```
**Note**: Set global variable DEVICE_ID to your camera's id.

Controls:
* 'c' - Capture, Warp, and Match
* 'q' - Quit
