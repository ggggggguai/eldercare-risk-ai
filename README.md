# YingLingAnJu ASR deployment

This branch contains the dedicated offline Chinese ASR service used by the
S10 three-task cognitive V3.5 pipeline. It exposes only:

```text
GET  /health/live
GET  /health/ready
POST /v1/asr/transcribe
```

The psychological algorithm service calls ASR over a private network using
`COGNITIVE_ASR_URL`. The backend does not call this service directly.

## Git and cloud assets

Git contains source, contracts, tests and deployment checks only. The frozen
`asr-paraformer-zh-v1.0` package (Paraformer, FSMN-VAD and CT-Punc) must be
restored from Tencent Cloud at:

```text
/app/models/asr/asr-paraformer-zh-v1.0
```

Every package file, including the internal `sha256sums.txt`, is bound by
`deploy/asr/asset_manifest.json`. The service fails closed before startup if
the external package, native FFmpeg runtime or API token is unavailable.

The repository keeps the Windows FFmpeg evidence in
`configs/runtime/asr-native-assets.json`. The Linux container generates
`asr-native-assets.container.json` during image construction and verifies the
same installed FFmpeg/FFprobe bytes on every startup. Windows executable
hashes are never reused as Linux evidence.

## Build and run

```text
docker build -t yinglinganju-asr .
docker run --gpus all --rm --name asr --network yinglinganju \
  -p 127.0.0.1:8011:8011 \
  --env-file /secure/asr.env \
  -v /srv/yinglinganju/asr-assets/models/asr:/app/models/asr:ro \
  yinglinganju-asr
```

Configure the psychological service with:

```text
COGNITIVE_ASR_URL=http://asr:8011/v1/asr/transcribe
```

Use one worker so one GPU loads one copy of the approximately 1.94 GiB frozen
model package. Real S10 audio remains an external device acceptance step.
