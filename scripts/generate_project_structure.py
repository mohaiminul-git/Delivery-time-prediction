"""Scaffolds an empty modular ML project skeleton, matching this repo's layout."""

import logging
import os
from pathlib import Path

while True:
    project_name = input("Enter your project name: ")
    if project_name != "":
        break

LIST_OF_FILES = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    "config/config.yml",
    "schema.yml",
    "app.py",
    "main.py",
    "setup.py",
]

for file_path in LIST_OF_FILES:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w") as f:
            pass
    else:
        logging.info(f"File already present at: {file_path}")
