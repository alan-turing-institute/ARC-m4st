# CallHome

## Pre-processing

Files:

```
align_lines.py
pipe.sh
remove_dialogue_delimiter.py
remove_noise_syntax.py
timestep_parsing.py
```

You would need to download all of the .cha transcript files from TalkBank [Spanish](https://ca.talkbank.org/access/CallHome/spa.html).

Processing the transcripts takes place in multiple stages.

The first stage is to align the transcripts to the English translations, and to remove the .cha specific syntax.

The second is to apply the official PERL pre-processing scripts distributed as part of the Fisher/Callhome Spanish extension dataset containing English translations.

You can use the ```pipe.sh``` script to do this, but make sure to set the paths to the dataset and the relevant scripts first.

## Experiments

Files:

```
main.py
fig8.py
fig7.py
```

The scripts ```fig8.py``` and ```metric_histograms.py``` produce the graph in Figure 8 in the report, and the coloured metric histograms in Figure 7.

Before running the figure scripts, you need to evaluate the metrics on the dataset by running ```main.py```. This will be slow. Example ```main.py``` run:

```
python main.py --text_folder ../../data/spa_processed --eng_folder ../../data/fisher_ch_spa-eng/data/corpus/ldc --mapping_folder ../../data/fisher_ch_spa-eng/data/mapping
```
