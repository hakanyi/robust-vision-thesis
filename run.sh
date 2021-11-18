#!/bin/bash

. load_config.sh

# Define the path to the container and conda env
CONT="${ENV['cont']}"
PYENV="${ENV['python']}"

# Parse the incoming command
COMMAND="$@"

# Enter the container and run the command
SING="${ENV['exec']} exec --nv"
mounts=(${ENV[mounts]})
BS=""
for i in "${mounts[@]}";do
    if [[ $i ]]; then
       BS="${BS} -B $i:$i"
    fi
done

$SING $BS $CONT bash -c "source $PWD/$PYENV/bin/activate \
	&& export PYTHON=$PWD/$PYENV/bin/python3 \
        && exec $COMMAND \
        && deactivate"
