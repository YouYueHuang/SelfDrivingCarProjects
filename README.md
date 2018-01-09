# SelfDrivingCarProjects

<img src="./img/cover.jpeg" alt="Overview" width="400px" height="267px">

## Overview

This repository is for Udacity self driving Nanodegree. The projects in this course uses the matplotlib, opencv, google tensorflow library. The working environment is based on python 3.6.1 virtualenv and jupyter notebook. 

##  Get started
------

* Download Anaconda 3.6 in official site
[Anaconda Distribution](https://www.anaconda.com/download/#windows)

* pip 
update pip, conda and conda-env to latest version
```python
conda install -c anaconda pip
```

* work in a virtual environment
create and activate a python virtual environment named tensorflow
```python
conda create -n tensorflow python=3.6.3
activate tensorflow
```
* Deactivate virtual environment
```python
deactivate
```

* Jupyter Notebook:
```python
pip install jupyter
```

* Launch an jupyter notebook inside a python virtualenv
```python
jupyter notebook
```

## Dependencies 
------

* tensorflow
```python
conda install -c conda-forge tensorflow
```
or
```python
pip install --ignore-installed --upgrade tensorflow
```

* install opencv
```python
conda install opencv
```
  > If the message 'ImportError: DLL load failed: The specified module could not be found' show, then download opencv binary file from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) In my case, I’ve used opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl. Go to the directory where the wheel file is located, and run the following command.
```python
pip install opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl
```

* matplotlib
```python
conda install -c conda-forge matplotlib 
```

* pillow

  > Pillow is a Python Imaging Library (PIL). Once Pillow is installed, the standard Matplotlib library can handel the figure of JPEG format and will not generate a ValueError any more. Otherwise 'ValueError: Format "jpg" is not supported' will show.
```python
pip install pillow
```

* moviepy 
```python
conda install -c conda-forge moviepy 
```

* pandas
```python
conda install -c anaconda pandas
```

* scikit-learn
```python
conda install -c anaconda scikit-learn 
```

* imageio
```python
conda install -c menpo imageio
```

* Matplotlib Jupyter Extension for the Matplotlib Jupyter widget
```python
conda install -c conda-forge ipympl 
conda install -c conda-forge widgetsnbextension
```
* seaborn
```python
conda install -c anaconda seaborn 
```

* plotly
```python
conda install -c anaconda plotly
```

* graphviz
  > This package facilitates the creation and rendering from Python.
```python
conda install -c conda-forge python-graphviz
```

## Reference
------
* [Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
* [Installing Jupyter with pip](http://jupyter.readthedocs.io/en/latest/install.html)
* [Downloading opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)
* [Google code guideline](https://google.github.io/styleguide/)
* [jupyter-matplotlib widget](https://github.com/matplotlib/jupyter-matplotlib)