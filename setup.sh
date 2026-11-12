#!/usr/bin/env bash
# Linux CPU setup. Run from the extracted starter directory.
set -euo pipefail
python3.12 -m venv .hw3-venv
.hw3-venv/bin/python -m pip install --upgrade pip
.hw3-venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.hw3-venv/bin/python -m pip install -r requirements-lock.txt
.hw3-venv/bin/python -m ipykernel install --user --name cs440-hw3 --display-name 'CS440 HW3 (PyTorch CPU)'
.hw3-venv/bin/python -c 'import sys,torch,numpy; print(sys.version,torch.__version__,numpy.__version__)'
