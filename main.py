from __future__ import annotations

"""
main.py

Lightweight environment and dependency sanity check. Prints key library versions
and whether CUDA is available for PyTorch. Useful before running preprocessing
or model training.
"""

import importlib
import sys
from typing import Dict, Tuple


def _try_import(pkg: str) -> Tuple[bool, str]:
    """Attempt to import a package and return (ok, version_or_reason)."""
    try:
        m = importlib.import_module(pkg)
        # Most packages expose __version__, but guard in case they don't
        ver = getattr(m, "__version__", "<no __version__>")
        return True, str(ver)
    except Exception as e:
        return False, f"Import failed: {e}"


def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check required dependencies and collect version info or failure reasons."""
    required = [
        "torch",
        "torchvision",
        "openl3",
        "librosa",
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "tqdm",
        "cv2",
        "moviepy",
        "soundfile",
    ]

    results: Dict[str, Tuple[bool, str]] = {}
    for name in required:
        pkg_name = name
        # Map import names that differ from pip names
        if name == "cv2":
            pkg_name = "cv2"
        if name == "sklearn":
            pkg_name = "sklearn"
        ok, info = _try_import(pkg_name)
        results[name] = (ok, info)
    return results


def print_report(results: Dict[str, Tuple[bool, str]]) -> None:
    print("\nDependency report:\n-------------------")
    for name, (ok, info) in results.items():
        status = "OK" if ok else "MISSING/ERROR"
        print(f"- {name:12s}: {status:13s} | {info}")

    # CUDA check for PyTorch
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_ok else 0
        print("\nPyTorch CUDA:")
        print(f"- CUDA available: {cuda_ok}")
        print(f"- GPU count     : {device_count}")
        if cuda_ok:
            print(f"- Current device: {torch.cuda.current_device()}")
            print(f"- Device name   : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"\nPyTorch CUDA check failed: {e}")


if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]}")
    res = check_dependencies()
    print_report(res)
    # Exit non-zero if any dependency failed to import (handy for CI)
    missing = [k for k, (ok, _) in res.items() if not ok]
    if missing:
        print("\nSome dependencies failed to import:", ", ".join(missing))
        sys.exit(1)
    sys.exit(0)
