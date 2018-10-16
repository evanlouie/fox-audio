#!/bin/bash
FOLDER="/Users/yvonneradsmikham/Microsoft/active-learning-detect/audio/wav/DP1_Clear_Gunshot_10s/*"
for filename in $FOLDER; do
    python white_noise_padding.py $filename
done