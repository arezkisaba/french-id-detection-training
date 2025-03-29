# Description

This project consists of training and creating a Tensorflow model based on Yolo11 to recognize certain official French documents, such as passports, identity cards, etc.  
The training was made using synthetic data.

# Setup

## Python3.11 virtual environment creation
```bash
cd ./src/scripts
python3.11 -m venv py_311
source py_311/bin/activate
python3.11 --version
```

## Dependencies installation
```bash
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
pip install ultralytics
pip install albumentations
pip install tensorflow
pip install tensorboard
pip install boto3
```

# Dataset generation & Training
## Openimages backgrounds download
```bash
python3.11 ./backgrounds-download.py
```
## Dataset generation
```bash
python3.11 ./dataset-generate.py
```
## Model training
```bash
python3.11 ./dataset-train.py
```
## Results
See results in ./results directory, models are located in ./results/weights
