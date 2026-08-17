# Open-source Checklist

Run this before pushing a public GitHub repository.

## Secrets

- Run `python scripts/check_secrets.py .`.
- Remove hard-coded `api_key`, `access_key`, `secret`, and `sk-...` values.
- Rotate any key that was ever committed or stored in local source.

## Data and Artifacts

- Confirm real datasets are not tracked.
- Confirm embeddings, checkpoints, pretrained weights, logs, and run outputs are not tracked.
- Confirm `.git`, `.idea`, `__pycache__`, and notebook caches are not copied into `baselines/`.

## Licenses

- Audit every copied baseline source tree.
- Keep third-party license files when available.
- Add attribution and citation notes for each paper implementation.
- If a method has no clear redistribution license, mark it before public release.

## Reproducibility

- Run `python -m unittest discover -s tests`.
- Run the toy SDSR evaluate command from the README.
- Run result aggregation.
- Confirm README commands work on a fresh clone.
