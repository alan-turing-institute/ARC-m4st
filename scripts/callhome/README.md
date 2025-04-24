# CallHome

This experiment used the CallHome dataset (informal conversations in Spanish with Spanish transcripts, and English translations) to measure the correlation between different translation metrics on more conversational data from an audio source.

## Dataset

Go [https://ca.talkbank.org/access/CallHome](here), select the conversation language, create account, then you can download the "media folder". There you can find the .cha files, which contain the transcriptions. You will need to pre-process this data in combination with the Callhome translations dataset, which includes part of the pre-processing scripts. The README under ./scripts/callhome of this repo contains more information.

To load the transcripts as an iterator, use the `m4st.callhome.pipeline.CallhomePipeline` class after pre-processing the data.

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
