# EEP595-Weather-Wizard
EEP595 Final Project Code Base

### Group Members:
* [Manpreet Singh](https://github.com/manpreet-singh)
* Mohammed Almousa
* [Wenzheng Zhao](https://github.com/Bobbed1999)
* Yuqi Nai

---

This repository contain the iPython notebook to train a AI/ML model on the [Historical Hourly Weather](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data) dataset available on kaggle. 

## Training the Model

Running the iPython notebook in it's entirety should read from the dataset, pull out the temperature, pressure, and humidity data for three cities close to Seattle from the full dataset.

Once the training and validation datasets have been created, the notebook trains the Neural Network on the dataset. Once the model has gone through training, it's then quantized and the performance of the quantized model is evaluated. 

Lastly, this notebook outputs both a tflite file and a C header file containing the model as an array of hex values, ready to be used with tflite and TinyML respectively.

## Arduino  Code

As for details about the Arduino code, you will need to install the following libraries to be able to run the model on a Arduino 33 BLE Sense (Rev2) board.
* EloquentTinyML - Provides an easy to use library to run inference on ML models
* Arduino_LPS22HB - Library to read from the Pressure Sensor
* Arduino_HS300x - Library to read from the Humidity Sensor 

The Arduino sketch in this repository will read the model from the [model.h](model.h) header file and will run inference on the model.