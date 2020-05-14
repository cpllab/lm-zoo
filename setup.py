from setuptools import setup


# Source version from package source
import re
version_file = "lm_zoo/__init__.py"
version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                          open(version_file).read(), re.M)
if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))


setup(
    install_requires=[
        "certifi==2020.4.5.1",
        "chardet==3.0.4",
        "click==7.1.2",
        "colorama==0.4.3",
        "crayons==0.3.0",
        "docker==4.2.0",
        "h5py==2.10.0",
        "idna==2.9",
        "numpy==1.18.4",
        "pandas==1.0.3",
        "python-dateutil==2.8.1",
        "pytz==2020.1",
        "requests==2.23.0",
        "six==1.14.0",
        "tqdm==4.46.0",
        "urllib3==1.25.9",
        "websocket-client==0.57.0",
    ],
    name="lm-zoo",
    packages=["lm_zoo"],
    scripts=["bin/lm-zoo"],
    version="1.1b0",
    python_requires=">=3.6",
    license="MIT",
    description="Command-line interface with state-of-the-art neural network language models",
    author="Jon Gauthier",
    author_email="jon@gauthiers.net",
    url="https://cpllab.github.io/lm-zoo",
    keywords=["language models", "nlp", "ai"],
)
