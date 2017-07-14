# Emojify

A web service that tries to determine the appropriate emoji for a given image. It uses a CNN model trained with the CIFAR-10 dataset deployed in ML Engine to run predictions.

## Setup

```sh
make
source venv/bin/activate
export PROJECT_ID=<Google Cloud Project ID>
export MODEL_NAME=<ML Engine Model Name>
```

## Run

```sh
# Dev
python emojify.py
# Prod
gunicorn emojify -b 0.0.0.0:8000
```
