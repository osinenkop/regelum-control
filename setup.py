from setuptools import setup, find_packages
from os.path import join, dirname
import sys, os

sys.path.insert(0, os.path.abspath(__file__ + "/.."))
with open(os.path.abspath(__file__ + "/../regelum/__init__.py"), "r") as f:
    for line in f.readlines():
        if "__version__" in line:
            exec(line)
            break
setup(
    name="regelum",
    version=__version__,
    author="AIDynamicAction",
    description="regelum is a framework for hybrid agent-environment loop simulation, with a library of predictive and stabilizing reinforcement learning setups",
    url="https://gitflic.ru/project/aidynamicaction/rcognita",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    long_description=open(join(dirname(__file__), "README.rst")).read(),
    long_description_content_type="text/x-rst",
    install_requires=[
        "matplotlib == 3.6.3",
        "mpldatacursor-rcognita == 0.7.2",
        "numpy >= 1.20.1",
        "scipy >= 1.5.0",
        "svgpath2mpl == 0.2.1",
        "tabulate >= 0.8.7",
        "systems == 0.1.0",
        "shapely == 1.7.1",
        "tabulate==0.8.7",
        "recursive_monkey_patch==0.4.0",
        "omegaconf==2.3.0",
        "hydra-core==1.3.2",
        "hydra-joblib-launcher==1.2.0",
        "pandas==1.4.0",
        "casadi>=3.5.5",
        "dill==0.3.7",
        "plotly==5.13.0",
        "gitpython==3.1.30",
        "filelock==3.0.12",
        "streamlit==1.17.0",
        "tables==3.8.0",
        "mlflow==2.4.1",
        "matplotx==0.3.10",
        "svgpathtools==1.6.1",
        "PyQt5==5.15.9",
    ],
    extras_require={
        "CASADI": ["casadi>=3.5.5"],
        "TORCH": ["torch>=1.13.1", "gpytorch==1.8.0"],
    },
    python_requires=">=3.6",
)
