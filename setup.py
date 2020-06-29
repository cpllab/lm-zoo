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
        "certifi==2020.6.20",
        "chardet==3.0.4",
        "click==7.1.2",
        "colorama==0.4.3",
        "crayons==0.3.1",
        "docker==4.2.1",
        "h5py==2.10.0",
        "idna==2.10",
        "numpy==1.19.0",
        "pandas==1.0.5",
        "python-dateutil==2.8.1",
        "pytz==2020.1",
        "requests==2.24.0",
        "semver==2.10.2",
        "six==1.15.0",
        "spython==0.0.84",
        "tqdm==4.47.0",
        "urllib3==1.25.9",
        "websocket-client==0.57.0",
    ],
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
