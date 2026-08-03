"""
Setup script for Ekegusii-LLM-Translation package.
Enables local installation in editable mode: `pip install -e .`
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="ekegusii_llm_translation",
    version="1.0.0",
    author="Aykahsay Research Team",
    author_email="research@ekegusii-nlp.org",
    description="Resource-Aware Adaptation of Multilingual LLMs for Ekegusii Machine Translation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aykahsay/Ekegusii-LLM-Translation",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ekegusii-nmt=src.cli.main:app",
        ],
    },
    include_package_data=True,
)
