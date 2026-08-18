"""Bounded, causal storage for environment-node readings.

The store deliberately lives in the algorithm process.  It does not infer
anything from a stale or future reading: callers must ask for a snapshot at a
frame receive time and the store only returns a causally preceding record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import threading
import time
from typing import Any, Mapping


class EnvironmentStoreError(ValueError):
    """A reading is not valid for the environment contract."""


class EnvironmentConflictError(RuntimeError):
    """A reading conflicts with an already accepted device sequence."""


@dataclass(frozen=True)
class EnvironmentRecord:
    schema_version: str
    device_id: str
    boot_id: str
    boot_started_at: datetime
    sequence: int
    observed_at: datetime
    clock_status: str
    illumination_lux: float | None
    illumination_status: str
    water_probes: dict[str, dict[str, str]]
    sensor_status: str
    received_at: datetime
    received_monotonic_sec: float
    transport_age_sec: float | None
    payload_sha256: str

    def to_dict(self, *, include_internal: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "device_id": self.device_id,
            "boot_id": self.boot_id,
            "boot_started_at": self.boot_started_at.isoformat(),
            "sequence": self.sequence,
            "observed_at": self.observed_at.isoformat(),
            "clock_status": self.clock_status,
            "illumination_lux": self.illumination_lux,
            "illumination_status": self.illumination_status,
            "water_probes": {name: dict(probe) for name, probe in self.water_probes.items()},
            "sensor_status": self.sensor_status,
            "received_at": self.received_at.isoformat(),
            "transport_age_sec": self.transport_age_sec,
        }
        if include_internal:
            value.update(
                {
                    "received_monotonic_sec": self.received_monotonic_sec,
                    "payload_sha256": self.payload_sha256,
                }
            )
        return value


@dataclass(frozen=True)
class EnvironmentSnapshot:
    status: str
    reason: str | None = None
    record: EnvironmentRecord | None = None
    snapshot_age_sec: float | None = None
    transport_age_sec: float | None = None

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.record is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "snapshot_age_sec": self.snapshot_age_sec,
            "transport_age_sec": self.transport_age_sec,
            "record": self.record.to_dict() if self.record else None,
        }


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise EnvironmentStoreError(f"{field} must be an ISO-8601 datetime") from exc
    else:
        raise EnvironmentStoreError(f"{field} is required")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EnvironmentStoreError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _finite_nonnegative(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EnvironmentStoreError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise EnvironmentStoreError(f"{field} must be a finite non-negative number")
    return number


def _normalise_water_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    if state in {"wet", "water", "flood", "alarm"}:
        return "water"
    if state in {"dry", "unknown"}:
        return state
    raise EnvironmentStoreError("water probe state must be dry, water or unknown")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EnvironmentStore:
    """Thread-safe per-device cache with idempotent sequence handling."""

    def __init__(self, *, capacity_per_device: int = 256) -> None:
        if int(capacity_per_device) < 1:
            raise ValueError("capacity_per_device must be positive")
        self.capacity_per_device = int(capacity_per_device)
        self._records: dict[str, deque[EnvironmentRecord]] = {}
        self._by_key: dict[tuple[str, str, int], EnvironmentRecord] = {}
        self._active_boot: dict[str, tuple[str, datetime, int]] = {}
        self._counters = {
            "accepted": 0,
            "duplicate": 0,
            "conflict": 0,
            "evicted": 0,
        }
        self._lock = threading.RLock()

    def ingest(
        self,
        payload: Mapping[str, Any],
        *,
        received_at: datetime | None = None,
        received_monotonic_sec: float | None = None,
        transport_age_sec: float | None = None,
        allow_backend_compatibility: bool = True,
    ) -> tuple[EnvironmentRecord, str]:
        """Validate and insert one reading.

        Returns ``(record, status)`` where status is ``accepted`` or
        ``duplicate``.  Conflicting sequences and old boots raise
        :class:`EnvironmentConflictError`.
        """
        if not isinstance(payload, Mapping):
            raise EnvironmentStoreError("environment reading must be an object")
        data = dict(payload)
        now = datetime.now(timezone.utc) if received_at is None else _parse_datetime(received_at, field="received_at")
        received_mono = time.monotonic() if received_monotonic_sec is None else float(received_monotonic_sec)
        if not math.isfinite(received_mono):
            raise EnvironmentStoreError("received_monotonic_sec must be finite")

        device_id = str(data.get("device_id", "")).strip()
        boot_id = str(data.get("boot_id", "")).strip()
        if not device_id or len(device_id) > 64:
            raise EnvironmentStoreError("device_id is required and must be <= 64 characters")
        if not boot_id or len(boot_id) > 64:
            raise EnvironmentStoreError("boot_id is required and must be <= 64 characters")
        try:
            sequence = int(data.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise EnvironmentStoreError("sequence must be a positive integer") from exc
        if sequence < 1:
            raise EnvironmentStoreError("sequence must be a positive integer")

        schema_version = str(data.get("schema_version", "1.0")).strip()
        if schema_version != "1.0":
            raise EnvironmentStoreError("unsupported environment schema_version")
        observed_value = data.get("observed_at") or data.get("received_at")
        boot_started_value = data.get("boot_started_at") or observed_value
        if allow_backend_compatibility and observed_value is None:
            observed_value = now
        if allow_backend_compatibility and boot_started_value is None:
            boot_started_value = now
        observed_at = _parse_datetime(observed_value, field="observed_at")
        boot_started_at = _parse_datetime(boot_started_value, field="boot_started_at")
        clock_status = str(data.get("clock_status", "synced")).strip().lower()
        if clock_status not in {"synced", "unsynced"}:
            raise EnvironmentStoreError("clock_status must be synced or unsynced")

        illumination_lux = _finite_nonnegative(data.get("illumination_lux"), field="illumination_lux")
        illumination_status = str(data.get("illumination_status", "fault")).strip().lower()
        if illumination_status not in {"ok", "degraded", "fault"}:
            raise EnvironmentStoreError("illumination_status must be ok, degraded or fault")

        raw_probes = data.get("water_probes")
        if not isinstance(raw_probes, Mapping) or not raw_probes:
            # GET responses from the backend expose an aggregate water_state;
            # retain it as a synthetic probe so the algorithm can still use it.
            aggregate = data.get("water_state")
            if aggregate is None:
                raise EnvironmentStoreError("water_probes must contain at least one probe")
            raw_probes = {"_aggregate": {"state": aggregate, "sensor_status": "ok"}}
        water_probes: dict[str, dict[str, str]] = {}
        for name, raw_probe in raw_probes.items():
            probe_name = str(name).strip()
            if not probe_name or len(probe_name) > 64 or not isinstance(raw_probe, Mapping):
                raise EnvironmentStoreError("water probe names and values are invalid")
            state = _normalise_water_state(raw_probe.get("state"))
            sensor_status = str(raw_probe.get("sensor_status", data.get("sensor_status", "ok"))).strip().lower()
            if sensor_status not in {"ok", "degraded", "fault"}:
                raise EnvironmentStoreError("water probe sensor_status must be ok, degraded or fault")
            water_probes[probe_name] = {"state": state, "sensor_status": sensor_status}

        sensor_status = str(data.get("sensor_status", "ok")).strip().lower()
        if sensor_status not in {"ok", "degraded", "fault"}:
            raise EnvironmentStoreError("sensor_status must be ok, degraded or fault")
        transport = transport_age_sec
        if transport is None and data.get("transport_age_sec") is not None:
            transport = _finite_nonnegative(data.get("transport_age_sec"), field="transport_age_sec")
        elif transport is not None:
            transport = _finite_nonnegative(transport, field="transport_age_sec")
        if transport is None:
            transport = max(0.0, (now - observed_at).total_seconds())

        record_payload = {
            "schema_version": schema_version,
            "device_id": device_id,
            "boot_id": boot_id,
            "boot_started_at": boot_started_at.isoformat(),
            "sequence": sequence,
            "observed_at": observed_at.isoformat(),
            "clock_status": clock_status,
            "illumination_lux": illumination_lux,
            "illumination_status": illumination_status,
            "water_probes": water_probes,
            "sensor_status": sensor_status,
        }
        payload_hash = _canonical_hash(record_payload)
        record = EnvironmentRecord(
            schema_version=record_payload["schema_version"],
            device_id=device_id,
            boot_id=boot_id,
            boot_started_at=boot_started_at,
            sequence=sequence,
            observed_at=observed_at,
            clock_status=clock_status,
            illumination_lux=illumination_lux,
            illumination_status=illumination_status,
            water_probes=water_probes,
            sensor_status=sensor_status,
            received_at=now,
            received_monotonic_sec=received_mono,
            transport_age_sec=transport,
            payload_sha256=payload_hash,
        )

        key = (device_id, boot_id, sequence)
        with self._lock:
            existing = self._by_key.get(key)
            if existing is not None:
                if existing.payload_sha256 == payload_hash:
                    self._counters["duplicate"] += 1
                    return existing, "duplicate"
                self._counters["conflict"] += 1
                raise EnvironmentConflictError(
                    "same device_id, boot_id and sequence has a different payload"
                )

            active = self._active_boot.get(device_id)
            if active is not None:
                active_boot_id, active_started, active_sequence = active
                if boot_started_at < active_started:
                    self._counters["conflict"] += 1
                    raise EnvironmentConflictError("reading belongs to an older boot")
                if boot_started_at == active_started and boot_id != active_boot_id:
                    self._counters["conflict"] += 1
                    raise EnvironmentConflictError("boot_id conflicts with active boot")
                if boot_id == active_boot_id and sequence <= active_sequence:
                    self._counters["conflict"] += 1
                    raise EnvironmentConflictError("sequence must increase within a boot")
                if boot_started_at > active_started:
                    self._active_boot[device_id] = (boot_id, boot_started_at, sequence)
                else:
                    self._active_boot[device_id] = (active_boot_id, active_started, sequence)
            else:
                self._active_boot[device_id] = (boot_id, boot_started_at, sequence)

            bucket = self._records.setdefault(device_id, deque())
            bucket.append(record)
            self._by_key[key] = record
            self._counters["accepted"] += 1
            while len(bucket) > self.capacity_per_device:
                removed = bucket.popleft()
                self._by_key.pop((removed.device_id, removed.boot_id, removed.sequence), None)
                self._counters["evicted"] += 1
            return record, "accepted"

    def snapshot_for_frame(
        self,
        device_id: str,
        *,
        frame_received_monotonic_sec: float,
        max_age_sec: float = 3.0,
        max_transport_age_sec: float = 2.0,
        max_future_skew_sec: float = 0.5,
    ) -> EnvironmentSnapshot:
        if not device_id:
            return EnvironmentSnapshot(status="unavailable", reason="environment_device_unbound")
        frame_time = float(frame_received_monotonic_sec)
        with self._lock:
            records = list(self._records.get(device_id, ()))
        candidates = [record for record in records if record.received_monotonic_sec <= frame_time]
        if not candidates:
            return EnvironmentSnapshot(status="unavailable", reason="causal_snapshot_missing")
        record = max(candidates, key=lambda item: item.received_monotonic_sec)
        age = frame_time - record.received_monotonic_sec
        if age < -float(max_future_skew_sec):
            return EnvironmentSnapshot(status="unavailable", reason="snapshot_from_future")
        if age > float(max_age_sec):
            return EnvironmentSnapshot(
                status="unavailable", reason="snapshot_expired", record=record,
                snapshot_age_sec=round(age, 4), transport_age_sec=record.transport_age_sec,
            )
        if record.clock_status != "synced":
            return EnvironmentSnapshot(
                status="unavailable", reason="device_clock_unsynced", record=record,
                snapshot_age_sec=round(age, 4), transport_age_sec=record.transport_age_sec,
            )
        if (record.observed_at - record.received_at).total_seconds() > float(max_future_skew_sec):
            return EnvironmentSnapshot(
                status="unavailable", reason="observed_time_in_future", record=record,
                snapshot_age_sec=round(age, 4), transport_age_sec=record.transport_age_sec,
            )
        if record.transport_age_sec is not None and record.transport_age_sec > float(max_transport_age_sec):
            return EnvironmentSnapshot(
                status="unavailable", reason="transport_age_exceeded", record=record,
                snapshot_age_sec=round(age, 4), transport_age_sec=record.transport_age_sec,
            )
        return EnvironmentSnapshot(
            status="valid", record=record,
            snapshot_age_sec=round(max(0.0, age), 4), transport_age_sec=record.transport_age_sec,
        )

    def readings(self, *, device_id: str | None = None, limit: int = 100, order: str = "desc") -> list[EnvironmentRecord]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            if device_id is None:
                values = [record for bucket in self._records.values() for record in bucket]
            else:
                values = list(self._records.get(device_id, ()))
        values.sort(key=lambda item: (item.received_at, item.received_monotonic_sec), reverse=(order != "asc"))
        return values[: int(limit)]

    def latest(self, device_id: str | None = None) -> EnvironmentRecord | None:
        values = self.readings(device_id=device_id, limit=1)
        return values[0] if values else None

    def snapshot_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "device_count": len(self._records),
                "record_count": sum(len(bucket) for bucket in self._records.values()),
                "capacity_per_device": self.capacity_per_device,
                "devices": {device: len(bucket) for device, bucket in self._records.items()},
                **self._counters,
            }
