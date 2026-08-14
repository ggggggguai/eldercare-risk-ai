# Anonymous collection field template — intentionally invalid

`template_only=true`. This document is not a real collection manifest. The JSON below is deliberately rejected by the formal validator because `dataset_role=template_only` and all external facts use invalid `<OWNER_REQUIRED>` placeholders.

The responsible owner/R5 must create the real anonymous manifest outside Git after C0 governance is established. Do not add direct identity, credentials, URLs, or absolute media paths.

```json
{
  "schema_version": "wandering-camera-collection-v1",
  "collection_id": "<OWNER_REQUIRED>",
  "dataset_role": "template_only",
  "authorization_receipt_id": "<OWNER_REQUIRED>",
  "participants": [
    {"participant_id": "<OWNER_REQUIRED>"}
  ],
  "camera_setups": [
    {
      "camera_setup_id": "<OWNER_REQUIRED>",
      "setup_id": "<OWNER_REQUIRED>"
    }
  ],
  "sessions": [
    {
      "session_id": "<OWNER_REQUIRED>",
      "participant_id": "<OWNER_REQUIRED>",
      "source_group_id": "<OWNER_REQUIRED>",
      "camera_setup_ids": ["<OWNER_REQUIRED>"]
    }
  ],
  "sources": [
    {
      "source_video_id": "<OWNER_REQUIRED>",
      "session_id": "<OWNER_REQUIRED>",
      "source_group_id": "<OWNER_REQUIRED>",
      "camera_setup_id": "<OWNER_REQUIRED>",
      "device_id": "<OWNER_REQUIRED>",
      "setup_id": "<OWNER_REQUIRED>",
      "stream_epoch": "<OWNER_REQUIRED>",
      "tracking_ref": "<OWNER_REQUIRED_RELATIVE_POSIX_REF>",
      "media_sidecar_ref": "<OWNER_REQUIRED_RELATIVE_POSIX_REF>"
    }
  ],
  "tracklet_participant_bindings": [],
  "participant_present_intervals": [],
  "clock_alignments": []
}
```
