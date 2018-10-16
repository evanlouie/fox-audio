#!/bin/bash
FOLDER="/Users/yvonneradsmikham/Microsoft/fox-audio/wav/Gunshot/*"
#Iteration with name
#for filename in $FOLDER; do
#    mv "$filename" "${filename// /_}"
#done
#Iteration with padding
for filename in $FOLDER; do
    python white_noise_padding.py $filename
done