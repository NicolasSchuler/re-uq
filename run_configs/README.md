# Run Configs

Tracked example provider matrices used by `scripts/run_experiment_from_config.py` and `scripts/run_task3_verification_from_config.py`.

| File | Purpose |
| --- | --- |
| `full_matrix.example.json` | Example full provider/model matrix; copy to `current_run.json` and edit locally. |
| `instructor_matrix.example.json` | Example matrix that uses the Instructor structured-output path with Pydantic response models. |

Local working files match `current_run*.json` and are gitignored. They contain machine- and credential-specific settings (endpoints, concurrency, API keys via environment variables) and must not be committed.

Copy and edit:

```bash
cp run_configs/full_matrix.example.json run_configs/current_run.json
```

See `docs/reproduction.md` for full usage.
