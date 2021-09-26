#!/bin/bash
echo "Start predition for computer_bild data (Feb,2021)."
python src/models/predict.py  --input_dir reports/data/computer_bild/input --output_dir reports/data/computer_bild/output --run_id=92bd93c5895144558545f21c6e5b2e08
