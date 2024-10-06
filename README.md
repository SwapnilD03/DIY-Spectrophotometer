
# DIY Spectrophotometer for Transformer Oil Aging Analysis

Welcome to the repository for my **DIY Low-Cost Spectrophotometer** project, designed to analyze the aging process of transformer oil samples using **spectrophotometry** and **machine learning**.

This project aims to provide an affordable solution for monitoring the condition of transformer oil, which is crucial for the safe and efficient operation of electrical transformers. By employing a deep learning algorithm and remote sample image transfer using **Firebase**, the spectrophotometer offers a modern, cost-effective method for real-time oil analysis.

## Project Overview

### Features
- **Spectrophotometry**: Measures the absorption of light at different wavelengths to determine the aging state of the transformer oil.
- **Machine Learning Model**: A deep learning algorithm processes the spectrophotometric data to classify the condition of the oil.
- **Remote Monitoring**: Developed a Firebase app that transfers sample images remotely for analysis from anywhere in the world.
- **Low-Cost Hardware**: Designed to be affordable without compromising on accuracy, using **Raspberry Pi** and other accessible hardware.

### Tech Stack
- **Programming Languages**: Python
- **Libraries**: 
  - Machine Learning: TensorFlow, Keras
  - Data Analysis: NumPy, Pandas
- **Hardware**: Raspberry Pi, Diffraction Grating,Oil Sample Holder
- **Firebase**: Remote sample image transfer

### Objective
The primary goal is to reduce the cost of equipment used for transformer oil analysis while maintaining accuracy. Traditional methods involve expensive equipment, but this project offers a low-cost alternative using simple electronics and machine learning techniques.

## Getting Started

### Prerequisites
- **Hardware**: Light Source (LED or laser), Oil Sample Holder,Diffraction Grating,Oil Samples,Webcam
- **Software**: Python, TensorFlow, Keras, Firebase



### Hardware Setup
- Position the Light Source perfectly so that maximum light passes through the oil sample
- Connect the sensor(Webcam) to capture the light that passes through the oil sample and send the data to the software.

## Machine Learning Model
The deep learning model is trained to classify oil samples based on the spectral data. The model utilizes convolutional neural networks (CNNs) to identify features that indicate aging in transformer oil.

### Training the Model
- Collect spectral data from various oil samples at different aging stages.
- Preprocess the data using **NumPy** and **Pandas**.
- Train the model using **TensorFlow** and **Keras**.

### Predictions

-After training the model,test the accuracy of the algorithm by testing it on newer samples.
-The model was found to efficiently identify and categorize different stages of aging of transformer oils.
