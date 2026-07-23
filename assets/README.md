# `assets/` — local files for encoders that can't ship their weights

A few encoders load weights or embeddings that are too large or too
license-encumbered to bundle with CORAL. By convention you drop those files
under `assets/<EncoderName>/`, and the encoder's `build()` (the path the
CLI uses) finds them automatically — so `coral extract --extractor Eva` just
works once the files are in place. No flags, no code changes.

The `<EncoderName>` is the registry name exactly (case-sensitive): `Eva`,
`CA-MAE`, `VIRTUES`. This `assets/` directory is always at the CORAL repo
root — there is no configurable override.

A missing file raises a `FileNotFoundError` that names the exact path to
populate.

## What each encoder needs

| Encoder | Place under `assets/<name>/` | Where to get it |
| --- | --- | --- |
| **Eva** | `GenePT_embedding.pkl` (~909 MB) | Zenodo record `10833191` |
| **CA-MAE** | a checkpoint package dir containing `huggingface_mae.py` (its name must be a valid Python identifier — no hyphens) | Mahmood Lab CA-MAE HF checkpoint |
| **VIRTUES** | `model.safetensors` + `base_config.yaml`; optional `embeddings/` dir of precomputed `{UniProt}.pt` | VIRTUES release (see `docs/specs/sprint-19-virtues-extractor.md`) |

Example layout:

```
assets/
├── Eva/
│   └── GenePT_embedding.pkl
├── CA-MAE/
│   └── camae_checkpoint/        # dir with huggingface_mae.py, vit.py, ...
└── VIRTUES/
    ├── model.safetensors
    ├── base_config.yaml
    └── embeddings/              # optional: {UniProt}.pt files
```

The other encoders (KRONOS1, KRONOS2, UNI, DINOv2, mean_marker) load from a
public Hub and need nothing here.

> Files under this directory are git-ignored — only this README and the
> `.gitignore` are tracked.
