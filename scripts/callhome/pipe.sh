#!/bin/bash

# Directories; Requires that you've unpacked the Fisher dataset.
PATH_TO_FISHER="../../data/fisher_ch_spa-eng"
# Path to the .cha transcript files.
PATH_TO_CALLHOME_TEXT="../../data/spa/text"
# Where the processed files will be output.
OUTPUT_PROCESSED_TEXT="../../data/spa_processed"

# These should run in order.
SCRIPT1="./align_lines.py"
SCRIPT2="./remove_noise_syntax.py"
SCRIPT3="${PATH_TO_FISHER}/data/bin/strip_markup.pl"
SCRIPT4="${PATH_TO_FISHER}/data/bin/remove_punctuation.pl"

GET_TIMESTEPS_SCRIPT="./timestep_parsing.py"

# Ensure output directory exists
mkdir -p "$OUTPUT_PROCESSED_TEXT"

# Process each file in the input directory
for file in "$PATH_TO_CALLHOME_TEXT"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        python "$SCRIPT1" "$file" | python "$SCRIPT2" | perl "$SCRIPT3" | perl "$SCRIPT4" > "$OUTPUT_PROCESSED_TEXT/$filename"
        python "$SCRIPT1" "$file" | python "$GET_TIMESTEPS_SCRIPT" > "$OUTPUT_PROCESSED_TEXT/${filename}_timestamps.txt"
    fi
done
