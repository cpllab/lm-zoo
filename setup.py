from setuptools import setup
from setuptools import find_packages


# Source version from package source
import re

version_file = "lm_zoo/__init__.py"
version_match = re.search(
    r"^__version__ = ['\"]([^'\"]*)['\"]", open(version_file).read(), re.M
)
if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))


setup(
    install_requires=[
        "click~=8.0.3",
        "crayons~=0.4.0",
        "h5py~=3.6.0",
        "numpy~=1.21.4; platform_machine != 'aarch64' and platform_machine != 'arm64' and python_version < '3.10'",
        "pandas~=1.3.4",
        "requests~=2.26.0",
        "tqdm~=4.62.3",
        "urllib3~=1.26.7",
    ],
    extras_require={
        "docker": ["docker~=5.0.3"],
        "singularity": ["spython~=0.1.17"],
        "huggingface": ["transformers >= 4.5, < 5.0", "torch"],
    },
    name="lm-zoo",
    packages=find_packages(exclude=["test"]),
    scripts=["bin/lm-zoo"],
    version=version_string,
    python_requires=">=3.6",
    license="MIT",
    description="Command-line interface with state-of-the-art neural network language models",
    author="Jon Gauthier",
    author_email="jon@gauthiers.net",
    url="https://cpllab.github.io/lm-zoo",
    keywords=["language models", "nlp", "ai"],
)
