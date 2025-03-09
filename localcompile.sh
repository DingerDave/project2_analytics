#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# Update this file with instructions on how to compile your code
echo "This localcompile file is designed for Macos and uses the uv tool you can find out more and install uv here:
https://github.com/astral-sh/uv" 
uv venv p2_venv --python 3.9
source p2_venv/bin/activate
uv pip install -r requirements.txt
