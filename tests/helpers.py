"""Shared unittest fixture helpers (imported explicitly; not a pytest conftest).

These helpers build canonical fixture payloads used across multiple test
modules so the per-test setup stays focused on the values each test actually
exercises.
"""


def raw_record(
    item,
    *,
    task,
    parsed_json,
    run_id="r1",
    model="m1",
    sample_kind="deterministic",
    parse_status="ok",
):
    """Build a canonical raw completion record for a benchmark ``item``.

    Only the fields that tests vary are exposed as parameters; every other
    field is fixed to the canonical value the inline copies used.
    """
    record = {
        "run_id": run_id,
        "model": model,
        "host": "http://localhost:8000/v1",
        "task": task,
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "sample_index": 0,
        "sample_kind": sample_kind,
        "temperature": 0.0,
        "top_p": 1.0,
        "prompt_version": "v1",
        "raw_text": "",
        "parsed_json": parsed_json,
        "parse_status": parse_status,
        "latency_s": 0.1,
        "error": "",
    }
    if "template_id" in item:
        record["template_id"] = item["template_id"]
    return record
