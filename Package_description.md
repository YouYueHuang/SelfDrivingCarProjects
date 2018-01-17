# Package install instruction
------

<img src="./img/package_cover.jpg" alt="Overview" width="400px" height="267px">

* tensorflow 
  - `conda install -c conda-forge tensorflow`
  - Or `pip install --ignore-installed --upgrade tensorflow`

* install opencv 

  1. `conda install opencv`
  2. And then `pip install opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl`

  > If the message 'ImportError: DLL load failed: The specified module could not be found' show, then download opencv binary file from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv) In my case, I’ve used opencv_python-3.3.1+contrib-cp36-cp36m-win_amd64.whl. Go to the directory where the wheel file is located, and run the following command.

* matplotlib `conda install -c conda-forge matplotlib`

* pillow `pip install pillow`

  > Pillow is a Python Imaging Library (PIL). Once Pillow is installed, the standard Matplotlib library can handel the figure of JPEG format and will not generate a ValueError any more. Otherwise 'ValueError: Format "jpg" is not supported' will show.


* moviepy `conda install -c conda-forge moviepy`

* pandas `conda install -c anaconda pandas`

* scikit-learn `conda install -c anaconda scikit-learn `

* imageio `conda install -c menpo imageio`

* Matplotlib Jupyter Extension 
  1. `conda install -c conda-forge ipympl`
  2. And then `conda install -c conda-forge widgetsnbextension`

* seaborn `conda install -c anaconda seaborn`

* plotly `conda install -c anaconda plotly`

* graphviz `conda install -c conda-forge python-graphviz`
  > This package facilitates the creation and rendering from Python.

* scikit-image `conda install -c anaconda scikit-image`
  > Image processing routines for SciPy

* export packages `conda list -e > requirement.txt`
* install all packages `conda create --name <env> --requirement.txt`

## Reference
------
* [Installing TensorFlow on Windows](https://www.tensorflow.org/install/install_windows)
* [Installing Jupyter with pip](http://jupyter.readthedocs.io/en/latest/install.html)
* [Downloading opencv](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)
* [Google code guideline](https://google.github.io/styleguide/)
* [jupyter-matplotlib widget](https://github.com/matplotlib/jupyter-matplotlib)