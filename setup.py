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
        "cached-property==1.5.2; python_version < '3.8'",
        "certifi==2021.10.8",
        "charset-normalizer==2.0.9; python_version >= '3'",
        "click==8.0.3",
        "colorama==0.4.4",
        "crayons==0.4.0",
        "h5py==3.6.0",
        "idna==3.3; python_version >= '3'",
        "importlib-metadata==4.8.2; python_version < '3.8'",
        "numpy==1.21.4; platform_machine != 'aarch64' and platform_machine != 'arm64' and python_version < '3.10'",
        "pandas==1.3.4",
        "python-dateutil==2.8.2",
        "pytz==2021.3",
        "requests==2.26.0",
        "semver==2.13.0",
        "six==1.16.0",
        "tqdm==4.62.3",
        "typing-extensions==4.0.1; python_version < '3.8'",
        "urllib3==1.26.7",
        "websocket-client==1.2.3",
        "zipp==3.6.0",
    ],
    extras_require={
        "docker": ["docker==5.0.3"],
        "singularity": ["spython==0.1.17"],
        "huggingface": ["transformers~=4.12", "torch"],
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
