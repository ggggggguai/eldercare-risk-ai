# Owner checklist for real C0 + C1 handoff

## 给负责人的通俗说明

下一阶段确实需要一段真实 development 视频，但必须先完成授权和存储安排。最简单的第一批可以由一名自愿、知情同意的成年人拍摄；不要求老人，也不能偷拍老人、家属、工作人员或路人。

第一段工程视频建议：

- 普通 MP4，固定手机或摄像头，不手持移动；
- 一个人完整出现在画面内，尤其保证双脚和行走区域可见；
- 不录音，避免镜子、证件、屏幕、门牌和其他隐私内容；
- 连续拍摄建议 60–120 秒；模型窗口本身为 40 秒，多留一些跟踪余量；
- 可先做自然走动或来回走动，用于确认 tracking、QC 和模型 forward 能否跑通；
- 视频保存在 Git 和仓库以外的受控目录，只向执行 AI 提供精确路径和读取授权。

完整 development 数据以后再补 direct、pacing、lapping、random-like，以及找东西、打电话踱步、清洁、锻炼、搬运等有目的困难负例和连续普通活动。第一段视频只用于 engineering smoke，不产生准确率或真实老人效果结论。

## Stop conditions

Do not run the video preparation CLI until every C0 item below is supplied by the responsible owner and the receipt passes the repository validator. Do not place the real receipt, collection manifest, video, identity mapping, credentials, or generated C1 directory in Git.

The repository templates are deliberately invalid. They are field guides, not authorization and not executable fixtures. The execution AI must not replace `<OWNER_REQUIRED>` values, change `template_only` into approval, assert consent/governance, invent validity dates or scopes, or infer facts from filenames or video content.

## C0 owner inputs

- [ ] Responsible owner confirms lawful consent/approval for this development purpose.
- [ ] `approval_status=approved`, `active=true`, and the current time is inside the owner-issued validity interval.
- [ ] `purpose=camera_development` and `dataset_role=development`.
- [ ] `allowed_operations` includes both `prepare_session` and the future `run_development` smoke.
- [ ] Anonymous participant, session, camera-setup, and source-group scopes cover the complete collection manifest.
- [ ] Owner confirms deidentified storage, access control, retention/deletion, withdrawal handling, and `audio_policy=not_collected`.
- [ ] The receipt ID exactly matches the collection's `authorization_receipt_id`.

## Anonymous collection inputs

- [ ] Each participant has an anonymous `participant_id`; no names, faces, contact details, or direct identifiers appear.
- [ ] Each session maps one anonymous participant, one source group, and its legal camera setups.
- [ ] Each source has a unique `source_video_id` and exact session/group/camera-setup/device/setup/stream-epoch linkage.
- [ ] `tracking_ref`, `media_sidecar_ref`, and later `media_ref` are deidentified relative POSIX references, never URLs or local absolute paths.
- [ ] Clock-domain, tracklet binding, and presence rows are consistent if already available; their existence does not substitute for C2/C3 labeled-evaluation readiness.

## Video and operator facts

- [ ] One authorized, deidentified development video is stored outside Git with access controls and the owner-defined retention policy.
- [ ] The selected `source_video_id` is declared exactly once in the validated collection.
- [ ] `media_ref` is an operator-provided deidentified relative POSIX reference.
- [ ] `camera_motion_state` is explicitly recorded as `stable`, `moved`, or `not_checked`; the AI does not infer it.
- [ ] `deidentification_status` is explicitly recorded by the operator; the AI does not infer it.
- [ ] `capture_started_at` and `timezone` are both provided, or both intentionally null.
- [ ] The fixed-camera assumption is appropriate for this C1 preparation; otherwise stop and revise the authorized collection plan rather than changing the controller.
- [ ] The fresh output directory is outside Git and does not already exist.

## Run order

Use the project environment and supply only owner-approved external paths/facts:

```bash
conda run --no-capture-output -n eldercare-ai python \
  scripts/wandering/build_camera_tracking_pair.py \
  --project-root . \
  --receipt <OWNER_REQUIRED_EXTERNAL_RECEIPT_JSON> \
  --collection <OWNER_REQUIRED_EXTERNAL_COLLECTION_JSON> \
  --source-video-id <OWNER_REQUIRED_ANONYMOUS_SOURCE_VIDEO_ID> \
  --input-video <OWNER_REQUIRED_EXTERNAL_AUTHORIZED_VIDEO> \
  --media-ref <OWNER_REQUIRED_DEIDENTIFIED_RELATIVE_POSIX_REF> \
  --camera-motion-state <OWNER_REQUIRED_STABLE_MOVED_OR_NOT_CHECKED> \
  --deidentification-status <OWNER_REQUIRED_STATUS> \
  --output-dir <OWNER_REQUIRED_FRESH_OUTPUT_OUTSIDE_GIT>
```

Add `--capture-started-at` and `--timezone` together only when the owner supplies both. Do not add authorization, scope, evidence, model, threshold, test, fake-runtime, loader, skip, or bypass arguments; the CLI intentionally has none.

## Acceptance before M0-CAM-D

- [ ] CLI completes into one fresh directory containing `tracking.jsonl`, `media_sidecar.json`, and `preparation_summary.json`.
- [ ] Output sidecar scope matches the chosen collection source and its tracking hash matches the canonical tracking bytes.
- [ ] Receipt remains approved, active, unexpired, and covers `run_development` at smoke time.
- [ ] A responsible owner explicitly records C0 and C1 acceptance outside Git.
- [ ] The software-only `M0-CAM-PORTABLE` runtime-asset preflight has passed.

The first four checks allow `C0=true` and `C1=true` to be recorded. An unlabeled M0-CAM-D engineering smoke may start only after the additional `M0-CAM-PORTABLE` check also passes. Without C2/C3, do not report accuracy/F1/FAR or tune thresholds/episode policy.
