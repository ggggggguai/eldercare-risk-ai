from __future__ import annotations

import subprocess
import sys
import textwrap


def test_facial_affect_tooling_does_not_require_pyarrow() -> None:
    probe = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise AssertionError("facial-affect tooling imported optional pyarrow")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = guarded_import

        from elderly_monitoring.modules.mental_health.submodules.facial_affect_clue import (
            casme2_official_preprocess,
        )

        assert casme2_official_preprocess is not None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
