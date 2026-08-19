# W5D-01 verification

- development records: `48` (`B01=36`, `B02=12`)
- boundary truth files hashed: `48`
- selected: `closing-s012-d08`
- pooled gate satisfied: `false`
- B01 recall/F1/tIoU/coverage: `0.637931/0.627119/0.773573/0.982759`
- B02 recall/F1/tIoU/coverage: `0.900000/0.947368/0.735423/0.900000`
- hard-break crossing count: `0`
- finalist deterministic replays: `closing-s012-d08, combined-m4-d015-s010-d10-r025, opening-m4-d015`
- all primary misses retained in failures: `true`
- all split/merge diagnostics retained in failures: `true`
- human truth or labels modified: `false`
- historical evidence overwritten: `false`

Run with the project `eldercare-ai` conda environment using `scripts/wandering/tune_camera_episode_boundaries.py`.
