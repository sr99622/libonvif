#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

DIST_DIR = Path("dist")
print(DIST_DIR)

def python_tag():
    v = sys.version_info
    return f"cp{v.major}{v.minor}"

def run_pip(args):
    cmd = [sys.executable, "-m", "pip"] + args
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    tag = python_tag()
    print(f"Detected Python tag: {tag}")

    wheels = list(DIST_DIR.glob("*.whl"))

    # Universal wheels (py3-none-any)
    universal = [w for w in wheels if "py3-none-any" in w.name]

    # ABI-specific wheels
    abi_specific = [w for w in wheels if f"-{tag}-" in w.name]

    if not abi_specific and not universal:
        print("No compatible wheels found.")
        sys.exit(1)

    # Separate onvif_gui
    gui_wheels = [w for w in universal + abi_specific if w.name.startswith("onvif_gui")]
    other_wheels = [w for w in universal + abi_specific if w not in gui_wheels]

    # Install non-GUI wheels first
    for wheel in sorted(other_wheels):
        run_pip(["install", str(wheel)])

    # Install GUI last
    for wheel in gui_wheels:
        run_pip(["install", str(wheel)])

    print("Installation complete.")

if __name__ == "__main__":
    main()
