from __future__ import annotations

from collections.abc import Sequence

from elderly_monitoring.common.schemas import AlgorithmEvent, action_for_level


class EventFusionPipeline:
    """Merge module events without implementing business workflows."""

    model_version = "fusion-v0.1"

    def merge(self, events: Sequence[AlgorithmEvent]) -> AlgorithmEvent:
        if not events:
            raise ValueError("At least one event is required for fusion.")

        highest = max(events, key=lambda event: (event.risk_level, event.risk_score))
        factors: list[str] = []
        for event in events:
            factors.extend(f"{event.module}:{factor}" for factor in event.risk_factors)

        return AlgorithmEvent(
            module="fusion",
            device_id=highest.device_id,
            person_id=highest.person_id,
            timestamp=highest.timestamp,
            scene_region=highest.scene_region,
            risk_level=highest.risk_level,
            risk_score=max(event.risk_score for event in events),
            confidence=round(sum(event.confidence for event in events) / len(events), 4),
            trigger_event=f"highest:{highest.module}:{highest.trigger_event}",
            risk_factors=factors,
            recommended_action=action_for_level(highest.risk_level),
            evidence_window=highest.evidence_window,
            model_version=self.model_version,
            metadata={"source_modules": [event.module for event in events]},
        )
