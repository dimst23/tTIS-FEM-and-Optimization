#!/usr/bin/env bash

set -e

cd 'to model directory'

for file in *.poly; do 
    if [ -f "$file" ]; then 
        time 'tetgen executable path' -zpq1.8/0O4a30kNEFAV "$file"
    fi 
done
