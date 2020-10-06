#!/bin/sh

# run in terminal to make it executable: chmod a+x terminal_installation.sh 
# write in terminal to execute it: 

pip install -e .

python3 -m pip install ipykernel
python3 -m spacy download en