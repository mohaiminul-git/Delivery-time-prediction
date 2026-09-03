from typing import List

from setuptools import find_packages, setup

REQUIREMENTS_PATH = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    with open(REQUIREMENTS_PATH, "r") as f:
        requirements = [req.strip() for req in f.readlines()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements


PROJECT_NAME = "delivery-time-prediction"
VERSION = "1.0.0"
DESCRIPTION = "A modular, production-style ML pipeline that predicts food delivery times."

AUTHOR = "Mohaiminul Islam"
AUTHOR_EMAIL = "imniloy11@gmail.com"

setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements(),
)

