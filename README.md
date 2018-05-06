# SelfDrivingCarProjects

<!-- <img src="./img/cover.jpeg" alt="Overview" width="400px" height="267px"> -->

<table style="width:100%">
  <tr>
    <th>
      <p align="center">P1: Basic Lane Finding</p>
    </th>
    <th>
      <p align="center">P2: Traffic Signs Classification</p>
    </th>
    <th>
      <p align="center">P3: Behavioral Cloning</p>
    </th>
    <th>
      <p align="center">P4: Advanced Lane Finding</p>
    </th>
    <th>
      <p align="center">P5: Vehicle Detectioin</p>
    </th>
  </tr>
  <tr>
    <th>
      <p align="center">
           <a href="https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/P1-LaneLines"><img src=".//P1-LaneLines//img//P1_example.gif" alt="Overview" width="95%"></a>
      </p>
    </th>
        <th><p align="center">
           <a href="https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/P2-Traffic-Sign-Classifier"><img src=".//P2-Traffic-Sign-Classifier//img//top5_prediction_2.png" alt="Overview" width="95%"></a>
        </p>
    </th>
       <th><p align="center">
           <a href="https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/P3-Behavioral-Cloning"><img src=".//P3-Behavioral-Cloning//imgs//record_03.gif" alt="Overview" width="95%"></a>
        </p>
    </th>
        <th><p align="center">
           <a href="https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/P4-advance-lane-finding"><img src=".//P4-advance-lane-finding//imgs//perspective_view_project_video.gif" alt="Overview" width="95%"></a>
        </p>
    </th>
        <th><p align="center">
           <a href="https://github.com/YouYueHuang/SelfDrivingCarProjects/tree/master/P5-Vehicle-Detection"><img src=".//P5-Vehicle-Detection//img//02.gif" alt="Overview" width="95%"></a>
        </p>
    </th>
  </tr>
</table>

## Overview

This repository is for Udacity self driving Nanodegree. The projects in this course uses the matplotlib, opencv, google tensorflow library. The working environment is based on python 3.6.1 virtualenv and jupyter notebook. 

##  Get started
------

* Download Anaconda 3.6 in official site
[Anaconda Distribution](https://www.anaconda.com/download/#windows)

* pip `conda install -c anaconda pip`

  > update pip, conda and conda-env to latest version

* Python virtual environment
  
  - create a python virtual environment named tensorflow `conda create -n tensorflow python=3.6.3`

  - activate virtual environment `activate tensorflow`

  - Deactivate virtual environment `deactivate`

* Installing a different version of Python

  - To create the new environment for Python 3.6, run `conda create -n py36 python=3.6 anaconda`
  - Do the similar for Python 2.7, run `conda create -n py36 python=3.6 anaconda`
  - To check the verision, activate the virtual environment and run `python --version`

* Jupyter Notebook 

  - install Jupyter Notebook `pip install jupyter`

  - Launch an jupyter notebook server `jupyter notebook`

* For behavior cloning,
  
  - conda install -c conda-forge eventlet
  - conda install -c conda-forge python-socketio

## Dependencies 
------
* export packages `conda list -e > requirement.txt`
* install all packages `conda create --name <env> --requirement.txt`

If you would like to install packages separately, this [link](Package_description.md) would help you. 