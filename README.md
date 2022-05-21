# House Price Prediction with DNN
House Price Prediction application with neural network. It also has a GUI which is created with PyQt.

## Cloning Repo

Go to your /home directory to install repo in it. It is important to be in **/home** directory.

```
cd ~
```


Then open terminal and paste the below command.

```
git clone https://github.com/farukbaykara/house_price_prediction_DNN.git
```

## Requirements

* Tensorflow 2.6
* Numpy
* Pandas
* Ipython
* Keras-Tuner
* scikit-learn
* PyQt5

## Installation

Requirements.txt and install.sh files have been created so that the requirements can be downloaded automatically. The bash script file will download the necessary libraries. Paste the code below into the terminal.
```
cd house_price_prediction_DNN
```

İnstallation of all packages can take a while. 

```
bash install.sh
```
## Running Program

If all the requirements have been downloaded, the interface is ready to run. Run the codes below in order.

```
cd house_price_pred_gui
```
```
python3 hpp_gui.py
```
## Using GUI

After launching interface window, write suitable house features to the inputs. Then, press the Set Input Params button and select a model type. Then
press the Predict button to see predicted house price. 