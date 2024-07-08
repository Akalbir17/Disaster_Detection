# Disaster Detection

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
The Disaster Detection project aims to develop a deep learning model for detecting and classifying disaster events from satellite imagery. The model is trained on the Comprehensive Disaster Dataset (CDD), which consists of images from various disaster categories such as floods, wildfires, and earthquakes, as well as non-disaster images.

## Dataset
The Comprehensive Disaster Dataset (CDD) is used for training and evaluating the disaster detection model. The dataset contains a large collection of satellite images categorized into different disaster types and non-disaster images. The images are stored in separate directories based on their respective categories.

The dataset is structured as follows:
Comprehensive Disaster Dataset(CDD)/
├── Damage_Earthquake/
├── Damage_Fire/
├── Damage_Flood/
├── Damage_Hurricane/
├── Damage_Tornado/
├── Non_Damage_Landslide/
├── Non_Damage_Misc/
├── Non_Damage_Smoke/
├── Non_Damage_Wildlife_Forest/
└── Non_damage_sea/

## Methodology

### Data Preprocessing
- The dataset is loaded from the specified directory using the `ImageDataGenerator` class from Keras.
- Data augmentation techniques such as rescaling, shear range, zoom range, and horizontal flip are applied to increase the diversity of the training data.
- The images are resized to a fixed size of 224x224 pixels.
- The dataset is split into training and validation sets using a specified validation split ratio.

### Model Architecture
- The disaster detection model is based on the VGG16 architecture, which is a deep convolutional neural network (CNN) pre-trained on the ImageNet dataset.
- The pre-trained VGG16 model is loaded without the top classification layers.
- Additional layers are added on top of the VGG16 base model, including a flatten layer, dense layers, and a final output layer with softmax activation for multi-class classification.

### Training
- The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
- Early stopping and model checkpointing are used to prevent overfitting and save the best model during training.
- The model is trained for a specified number of epochs with a batch size of 32.

### Evaluation
- The trained model is evaluated on the validation set to assess its performance.
- Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated.
- A classification report and confusion matrix are generated to provide detailed insights into the model's performance for each disaster category.

## Results
The disaster detection model achieved an accuracy of 86.7% on the validation set. The classification report and confusion matrix provide further insights into the model's performance for each disaster category.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Usage
1. Clone the repository: git clone https://github.com/Akalbir17/Disaster_Detection.git

2. Install the required dependencies: pip install -r requirements.txt

## License
This project is licensed under the [MIT License](LICENSE).
