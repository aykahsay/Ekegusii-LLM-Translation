from setuptools import setup, find_packages

setup(
    name="ekegusii_llm_translation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "peft>=0.8.0",
        "datasets>=2.16.0",
        "sacrebleu>=2.4.0",
        "pandas>=2.1.0"
    ],
    author="Aykahsay",
    description="Resource-Aware Adaptation of Multilingual LLMs for Ekegusii Translation",
)
