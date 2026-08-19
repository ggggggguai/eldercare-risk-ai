# W5D-05 camera algorithm handoff

Run ID: `w5d05-b01-b02-handoff-20260818-v1`

This directory is the backend-consumable `module=mental_health` evidence bundle.
It preserves episode shape, optional context, person-day aggregation, rolling
baseline readiness, unavailable/error states, source identities, and hashes.

## Reproduce the complete cached-tracking flow

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_handoff_e2e.py --output-root tmp/wandering_camera_5d_runs/<fresh_run_id> --run-id <fresh_run_id>
```

The command starts from the registered B01+B02 tracking/sidecar index, then runs
automatic episode/shape, optional three-frame context, daily/baseline, and final
handoff assembly without manual JSON edits. The output root must not exist.

## Contract

Every JSONL row carries `schema_version`, `module`, stable record identity,
status/quality, model/config/policy identity, and source references. The manifest
binds schema IDs, counts, byte sizes, SHA-256 values, source stages, and home-smoke
state. `algorithm_event_emitted=false`; no diagnosis, business risk decision,
backend, or frontend is implemented here.

## Current evidence limit

This is B01+B02 development evidence. Capture wall-clock timestamps were absent,
so the daily binding uses an explicitly declared schedule. Home input is still
awaiting input and this bundle is not `home_validated`; deterministic fake context
only verifies the adapter contract and is not context-label accuracy evidence.
