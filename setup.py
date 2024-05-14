import os
import re
import subprocess
from pathlib import Path

from setuptools import setup, find_packages

from gemlib.version import __version__

setup(
    name="gemlib",
    packages=find_packages(),
    entry_points={
        "console_scripts": ['gemlib = gemlib.gemlibmain:main']
    },
    version=__version__,
    description="Generic Engine of Machine Learning",
    url="<Confluence Page TBA>",
    install_requires=[
        # packages required for running epas go here, to be installed from wherever
        # this is different to requirements.txt, see here:
        # https://caremad.io/posts/2013/07/setup-vs-requirement/
        # https://www.python.org/dev/peps/pep-0440/#version-specifiers
        "transformers==4.9.2",
        "sentence-transformers==2.0.0",
        "gensim==3.8.3",
        "spacy==2.3.4"
        #"numpy>=1.14.3",
        #"pandas>=0.23.0",
        #"pyarrow>=0.8.0",
        #"pyyaml>=3.12",
        #"tables>=3.4.3",
        #"scikit-learn>=0.19.1",
        #"scipy>=1.0.0",
        #"seaborn>=0.8.1",
        #"pydotplus>=2.0",
        #"networkx>2.1",
        #"tqdm>4.19.5",
        #"redis>2.10.6"
    ]
)
