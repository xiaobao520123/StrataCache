from __future__ import annotations

import glob
import os
import runpy
import sys
from typing import Callable


def _iter_test_files() -> list[str]:
    here = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(here, "test_*.py")))
    return files


def main(argv: list[str]) -> int:
    _ = argv
    failures: list[str] = []
    for path in _iter_test_files():
        ns = runpy.run_path(path)
        run: Callable[[], None] | None = ns.get("run")  # type: ignore[assignment]
        if run is None:
            print(f"[SKIP] {os.path.basename(path)} (no run())")
            continue
        try:
            run()
            print(f"[ OK ] {os.path.basename(path)}")
        except Exception as e:  # noqa: BLE001
            failures.append(os.path.basename(path))
            print(f"[FAIL] {os.path.basename(path)}: {e}")
    if failures:
        print("Failures:", ", ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

