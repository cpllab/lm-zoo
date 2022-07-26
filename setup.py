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
        "certifi==2022.6.15; python_full_version >= '3.6.0'",
        "charset-normalizer==2.1.0; python_full_version >= '3.6.0'",
        'click~=8.1.3', "colorama==0.4.5; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        'crayons==0.4.0',
        'h5py~=3.7.0',
        "idna==3.3; python_version >= '3.5'",
        "numpy~=1.23.1; python_version >= '3.8'",
        'pandas~=1.4.3',
        'python-dateutil==2.8.2',
        'pytz==2022.1',
        'requests==2.28.1',
        "semver==2.13.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "six==1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        'tqdm==4.64.0',
        "urllib3==1.26.11; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5' and python_version < '4'",
        "websocket-client~=1.3.3; python_version >= '3.7'"
    ],
    extras_require={
        "docker": ["docker~=5.0.3"],
        "singularity": ["spython~=0.2.1"],
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
