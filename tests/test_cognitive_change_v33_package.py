from __future__ import annotations

from training.cognitive_change_clue.package_model import MODEL_VERSION


def test_package_version_is_frozen() -> None:
    assert MODEL_VERSION == "cognitive-mm-v3.3.0"
