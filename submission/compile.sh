#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# Update this file with instructions on how to compile your code
echo "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv p2_venv --python 3.9
source p2_venv/bin/activate
uv pip install -r requirements.txt