#!/usr/bin/env python3

import json
import os
import torch


SPECS_FILE = "specs.json"
MESH_NAMES_FILE = "base_postures.txt"
STIMULI_DIR = "stimuli"
# file that holds the lamp angles to be used to create images
IMG_IN_FILE = "images_input.csv"
IMG_OUT_FILE = "images_output.csv"
VID_OUT_FILE = "movies_output.csv"
DATA_DIR = "data"
DB_FILE = "participants.db"
DATA_FILE = "parsed_trials.csv"

def load_experiment_specifications(experiment_directory):
    file_pth = os.path.join(experiment_directory, SPECS_FILE)

    with open(file_pth, "r") as f:
        specs = json.load(f)

    return specs

def load_mesh_names(experiment_directory):
    file_pth = os.path.join(experiment_directory, MESH_NAMES_FILE)

    with open(file_pth, "r") as f:
        stim_list = f.read().splitlines()

    return stim_list
