#####################
## Troubleshooting ##
#####################
Here you find some issues when you experience some trouble at installing or 
using this project.

##################
# Trouble issues #
##################
1.) some libs are not working
for example when you use one of the following
  python src/data/stage_resmed.py --input_path /path/ --output_path /path/ --station bia
  python src/data/make_dataset.py
  python src/models/train.py
  python src/models/predict.py --run_id=4d8ddb41e7f340c182a6a62699502d9f --score_file_pattern=202012*
  python src/visualization/explain.py

2.) having problems with hdbscan

############
# Solution #
############
1. Own conda environment
Create an own python3.7 conda environment with name "anomalia"
and install needed libraries by the follwing environment description:
environment_conda_anomalia_python3.7.yml ...

Commands:
> conda environment_conda_anomalia.yml
> conda env create -f environment_conda_anomalia_python3.7.yml

2. Install packages not available not supported by conda
afterwards also install libs that may not avaible by conda ...
Command:
> pip install -r requirements.txt

and here you go. Finished.

General guide lines working with conda environments:
[https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html]


