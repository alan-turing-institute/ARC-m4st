## CallHome plots

This folder contains the scripts:

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
