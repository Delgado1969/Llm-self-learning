#!/bin/bash

echo " Preparando dataset..."
python src/prepare_dataset.py

echo " Fine-tuning del modelo..."
python src/fine_tune.py

echo " Evaluando el modelo..."
python src/evaluate.py