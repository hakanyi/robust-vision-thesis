#!/bin/bash

# Loads config

if [ -f "user.conf" ]; then
    CFGFILE="user.conf"
else
    CFGFILE="default.conf" # change to "default.conf"
fi

while read line; do
    if [[ $line =~ ^"["(.+)"]"$ ]]; then
        arrname=${BASH_REMATCH[1]}
        declare -A $arrname
    elif [[ $line =~ ^([_[:alpha:]][_[:alnum:]]*)":"(.*) ]]; then
        declare ${arrname}[${BASH_REMATCH[1]}]="${BASH_REMATCH[2]}"
    fi
done < $CFGFILE
