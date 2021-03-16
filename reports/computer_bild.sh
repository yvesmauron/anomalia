#!/bin/bash
echo "Start predition for computer_bild data (Feb,2021)."
python src/models/predict.py  --input_dir reports/data/computer_bild/input --output_dir reports/data/computer_bild/output 
