# SelfDrivingCarProjects

<img src="./img/cover.jpeg" alt="Overview" width="400px" height="267px">

## Overview

This repository is for Udacity self driving Nanodegree. The projects in this course uses the matplotlib, opencv, google tensorflow library. The working environment is based on python 3.6.1 virtualenv and jupyter notebook. 

##  Get started
------

* Download Anaconda 3.6 in official site
[Anaconda Distribution](https://www.anaconda.com/download/#windows)

* pip `conda install -c anaconda pip`

  > update pip, conda and conda-env to latest version

* Work in a python virtual environment
  
  - create a python virtual environment named tensorflow 'conda create -n tensorflow python=3.6.3'

  - activate virtual environment `activate tensorflow`

  - Deactivate virtual environment `deactivate`

* Work with Jupyter Notebook 

  - install Jupyter Notebook `pip install jupyter`

  - Launch an jupyter notebook server `jupyter notebook`

## Dependencies 
------

* tensorflow `conda install -c conda-forge tensorflow` or `pip install --ignore-installed --upgrade tensorflow`

* install opencv `conda install opencv` and then `pip install opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl`

  > If the message 'ImportError: DLL load failed: The specified module could not be found' show, then download opencv binary file from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) In my case, I’ve used opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl. Go to the directory where the wheel file is located, and run the following command.

* matplotlib `conda install -c conda-forge matplotlib`

* pillow `pip install pillow`

  > Pillow is a Python Imaging Library (PIL). Once Pillow is installed, the standard Matplotlib library can handel the figure of JPEG format and will not generate a ValueError any more. Otherwise 'ValueError: Format "jpg" is not supported' will show.


* moviepy `conda install -c conda-forge moviepy `

* pandas `conda install -c anaconda pandas`

* scikit-learn `conda install -c anaconda scikit-learn `

* imageio `conda install -c menpo imageio`

* Matplotlib Jupyter Extension `conda install -c conda-forge ipympl` and then `conda install -c conda-forge widgetsnbextension`

* seaborn `conda install -c anaconda seaborn`

* plotly `conda install -c anaconda plotly`

* graphviz `conda install -c conda-forge python-graphviz`
  > This package facilitates the creation and rendering from Python.

## Reference
------
* [Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
* [Installing Jupyter with pip](http://jupyter.readthedocs.io/en/latest/install.html)
* [Downloading opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)
* [Google code guideline](https://google.github.io/styleguide/)
* [jupyter-matplotlib widget](https://github.com/matplotlib/jupyter-matplotlib)