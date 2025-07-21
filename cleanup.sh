#!/bin/bash

list=$(find . -type d -name "__pycache__" | grep -v ".venv")
for dir in $list; do
    echo "Removing $dir"
    rm -rf "$dir"
done

rm -rf **/_storage/*