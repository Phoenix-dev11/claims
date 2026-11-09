# claims-forge

Extraction package and consensus miner setup for the Claims competition (Bittensor subnet 111).

## Contents

- `src/claims_forge/`: the competition package.
- `dist/` and `competition_manifest.json`: the built wheel and its manifest.
- `deploy/`: consensus miner setup and start scripts.
- `eval/`: local scoring harness. It needs the `eval/.venv` environment and downloaded test papers.
- `tests/`: offline unit tests.

## Run the miner on a new machine

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and inbound TCP port 8091.
Copy only the hotkey and `coldkeypub.txt` to the server, never the coldkey.

```sh
deploy/setup.sh
# Fill in .env, which is never committed.
deploy/run-miner.sh --claims.dry-run
deploy/run-miner.sh
```

## Build the package

```sh
uv venv --python 3.12 .venv
uv pip install -e .
.venv/bin/python -m unittest discover -s tests
uv build --wheel --out-dir dist
```

Parts of this project derive from the MIT-licensed
[claims-miner-package](https://github.com/DeSciClaims/claims-miner-package); see `LICENSE`.
