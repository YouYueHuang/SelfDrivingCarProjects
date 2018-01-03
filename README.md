# SelfDrivingCarProjects

This repository is for Udacity self driving class. The projects in this course uses the matplotlib, opencv, google tensorflow library. I have created a python 3.6.1 virtualenv and have installed jupyter and tensorflow there. 

## Installation

Download Anaconda 3.6 in official site
(Anaconda Distribution)[https://www.anaconda.com/download/#windows]

------
update pip, conda and conda-env to latest version
```python
conda install -c anaconda pip
```

create and activate a python virtual environment named tensorflow
```python
C://Users//self_driving_car_projects>conda create -n tensorflow python=3.6.3
C://Users//self_driving_car_projects>activate tensorflow
```

install tensorflow
```python
(tensorflow) C://Users//self_driving_car_projects>conda install -c conda-forge tensorflow
```
or
```python
(tensorflow) C://Users//self_driving_car_projects>pip install --ignore-installed --upgrade tensorflow
```

install opencv
```python
(tensorflow) C://Users//self_driving_car_projects>conda install opencv
```
If the message 'ImportError: DLL load failed: The specified module could not be found' show, then download opencv binary file from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) In my case, I’ve used opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl. Go to the directory where the wheel file is located, and run the following command.
```python
(tensorflow) C://Users//self_driving_car_projects>pip install opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl
```

install matplotlib
```python
(tensorflow) C://Users//self_driving_car_projects>conda install -c conda-forge matplotlib 
```

install pillow

Pillow is a Python Imaging Library (PIL). Once Pillow is installed, the standard Matplotlib library can handel the figure of JPEG format and will not generate a ValueError any more. Otherwise 'ValueError: Format "jpg" is not supported' will show.
```python
(tensorflow) C://Users//self_driving_car_projects>pip install pillow
```

install moviepy 
```python
conda install -c conda-forge moviepy 
```

install pandas
```python
conda install -c anaconda pandas
```

install scikit-learn
```python
conda install -c anaconda scikit-learn 
```

install imageio
```python
conda install -c menpo imageio
```

install the Jupyter Notebook:
```python
(tensorflow) C://Users//self_driving_car_projects>pip install jupyter
```

Start an jupyter notebook inside a python virtualenv
```python
(tensorflow) C://Users//self_driving_car_projects>jupyter notebook
```

Install Matplotlib Jupyter Extension for the Matplotlib Jupyter widget
```python
(tensorflow) C://Users//self_driving_car_projects>conda install -c conda-forge ipympl 
(tensorflow) C://Users//self_driving_car_projects>conda install -c conda-forge widgetsnbextension
```
Install seaborn
```python
(tensorflow) C://Users//self_driving_car_projects>conda install -c anaconda seaborn 
```

Install plotly for interactive dashboard
```python
(tensorflow) C://Users//self_driving_car_projects>conda install -c anaconda plotly
```

Deactivate virtual environment
```python
(tensorflow) C://Users//self_driving_car_projects>deactivate tensorflow
```

## Reference
------
* [Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
* [Installing Jupyter with pip](http://jupyter.readthedocs.io/en/latest/install.html)
* [Downloading opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)
* [Google code guideline](https://google.github.io/styleguide/)
* [jupyter-matplotlib widget](https://github.com/matplotlib/jupyter-matplotlib)