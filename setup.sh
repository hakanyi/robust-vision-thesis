#!/bin/bash

. load_config.sh

usage="Syntax: $(basename "$0") [-h|--help] [COMPONENTS...] -- will set up the project environment,

where:
    -h | --help     Print this help
    COMPONENTS...   Specify component to set up

Valid COMPONENTS:
    all: set up all components (container will be pulled, not built)
    cont_[pull|build]: pull the singularity container or build it
    python: install the python environment via poetry
    stimuli: download stimuli (images and movies)
    data: pull behavioral data"

if [[ $# -eq 0 ]] || [[ "$@" =~ "--help" ]] || [[ "$@" =~ "-h" ]];then
    echo "$usage"
    exit 0
fi

EXP_DIR="experiments/mooney-bodies"

# container setup
if [[ "$@" =~ "cont_pull" ]] || [[ "$@" =~ "all" ]];then
    echo "Pulling singularity container..."
    wget "https://yale.box.com/shared/static/bj26wxp80rjkic9ns13tsbdh7h2mpnxs.sif" -O "${ENV[cont]}"
elif [[ "$@" =~ "cont_build" ]];then
    echo "Building singularity container..."
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity
else
    echo "Not touching container"
fi

# python env setup
if [[ "$@" =~ "python" ]] || [[ "$@" =~ "all" ]];then
    echo "Setting up python environment..."
    singularity exec ${ENV[cont]} bash -c "poetry install --no-dev"
else
    echo "Not touching python environment"
fi

# download stimuli
if [[ "$@" =~ "stimuli" ]] || [[ "$@" =~ "all" ]];then
    if [! -d "${EXP_DIR}/stimuli"]; then
        echo "Pulling stimuli"
        wget "https://yale.box.com/shared/static/dzdedwubwev7svdh7flsxraa2eyh99ik.gz" -O stimuli.tar.gz
        tar -xzvf stimuli.tar.gz -C "${EXP_DIR}/stimuli"&& rm stimuli.tar.gz
    else
        echo "Stimulus directory exists. Not pulling any."
    fi
else
    echo "Not pulling any stimuli"
fi

# download data
if [[ "$@" =~ "data" ]] || [[ "$@" =~ "all" ]];then
    if [! -d "${EXP_DIR}/data"]; then
        echo "Pulling data"
        wget "https://yale.box.com/shared/static/u86xcm05higxo5br1opj4tzqzepfi2zh.gz" -O data.tar.gz
        tar -xzvf data.tar.gz -C "${EXP_DIR}/data"&& rm data.tar.gz
    else
        echo "Data directory exists. Not pulling any."
    fi
else
    echo "Not pulling any data"
fi
