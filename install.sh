#!/bin/sh
sudo apt install python3-pip
pip install --upgrade pip
pip install -r requirements.txt
pip install future
pip install keras-tuner -q
pip install typing-extensions --upgrade
sudo apt-get install python3-pyqt5

