#!/bin/bash
script_path="main.py"
env_name='rnns'

# tell the shell to use the created virtual environment
source  ~/anaconda3/etc/profile.d/conda.sh
source activate $env_name
# install requirements
pip3 install -r requirements.txt
# download english model from spacy
python3 -m spacy download en_core_web_sm
# Run the script
python3 $script_path
# deactivate the virtual environment
source deactivate