# W5D-04 real daily reports and rolling personal baseline

- validation scope: `deterministic_replay`
- daily reports: `15`
- persons: `1`
- local dates: `15`
- warming_up / initial_ready / stable_ready: `3 / 4 / 8`
- home input: `awaiting_input`
- home smoke: `not_run_input_unavailable`

B01+B02 rows use real development episode/tracking evidence with explicit person/session/timezone/presence binding. Capture clocks were absent, so the binding manifest uses a declared development replay schedule and does not claim observed wall-clock time.

Deterministic replay bundles prove only the 3/7/14-day state machine. They are not real longitudinal observations, clinical validation, risk decisions, or AlgorithmEvent output.
