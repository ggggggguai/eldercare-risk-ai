# Psychological algorithm deployment

This branch deploys one psychological algorithm service with two product
branches:

1. mood/social: R11 current-state multi-expert orchestration (including the
   frozen B0 facial-affect clue where the R11 contract uses it) plus the
   independent V3.4 future-trend forecast;
2. cognitive change: frozen S10 three-task V3.5 plus wandering daily evidence
   and `cognitive-wandering-rule-fusion-v1`.

The service is separate from the already deployed fall-risk process.  The
wandering camera chain reuses selected `fall_risk` tracking source code, but
this image does not register fall monitoring session endpoints.

## Git and assets

Git contains source, contracts, configuration and deployment checks only.
Weights, feature statistics, runtime manifests, videos, datasets and experiment
reports must be stored in Tencent Cloud and restored under `/app` using the
exact paths in `asset_manifest.json`.  The entrypoint verifies every byte count
and SHA-256 before starting the API.  Credentials are environment variables and
must never be added to the repository.

The received Step5/Step6 handoff contains valid RF/TCN development artifacts,
but it does not contain the later camera-primary five-file bundle
`topowander_m0r_candidate_v3`.  Those five exact files remain a required cloud
asset and are listed with their frozen hashes.  Until they are supplied,
wandering camera runtime readiness must remain failed; daily handoff ingestion
and cognitive/wandering rule-fusion code are still fully versioned.

## Build and run

```text
docker build -f Dockerfile.mental-health -t yinglinganju-mental-health .
docker run --gpus all --rm -p 8080:8080 \
  --env-file /secure/mental-health.env \
  -v /srv/yinglinganju/assets/models:/app/models:ro \
  -v /srv/yinglinganju/assets/data:/app/data:ro \
  -v /srv/yinglinganju/assets/reports:/app/reports:ro \
  yinglinganju-mental-health
```

Use the backend `MENTAL_HEALTH_ALGORITHM_BASE_URL` to point at this service.
`COGNITIVE_ASR_URL` must point to the private ASR service.  Real S10 and home
wandering validation remain external acceptance work and are not represented
by local fixtures.
