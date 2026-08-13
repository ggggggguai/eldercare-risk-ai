# Authorization receipt field template — intentionally invalid

`template_only=true`. This document is not approval, consent, a signature, or an authorization receipt. The JSON below is deliberately rejected by the formal validator: approval is `template_only`, active is false, dates are invalid/expired, governance is false, and required facts use `<OWNER_REQUIRED>`.

The responsible owner must create and retain the real receipt outside Git. The execution AI must not fill or sign it.

```json
{
  "schema_version": "wandering-camera-authorization-receipt-v1",
  "receipt_id": "<OWNER_REQUIRED>",
  "approval_status": "template_only",
  "active": false,
  "purpose": "<OWNER_REQUIRED>",
  "dataset_role": "template_only",
  "valid_from": "1970-01-01T00:00:00+00:00",
  "expires_at": "1970-01-01T00:00:00+00:00",
  "allowed_operations": ["<OWNER_REQUIRED>"],
  "participant_ids": ["<OWNER_REQUIRED>"],
  "session_ids": ["<OWNER_REQUIRED>"],
  "camera_setup_ids": ["<OWNER_REQUIRED>"],
  "source_group_ids": ["<OWNER_REQUIRED>"],
  "governance": {
    "consent_confirmed": false,
    "deidentified_storage": false,
    "access_control_confirmed": false,
    "retention_and_deletion_defined": false,
    "withdrawal_process_defined": false,
    "audio_policy": "template_only"
  }
}
```
