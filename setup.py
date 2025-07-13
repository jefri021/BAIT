from setuptools import setup, find_packages

setup(
    name="bait",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "loguru",
        "tqdm",
        "openai",
        "ray",
    ],
    entry_points={
        'console_scripts': [
            'bait-scan=scripts.scan:main',
            'bait-eval=scripts.eval:main',
        ],
    },
    author="SolidShen",
    description="BAIT: LLM Backdoor Scanning Tool",
    python_requires=">=3.8",
)