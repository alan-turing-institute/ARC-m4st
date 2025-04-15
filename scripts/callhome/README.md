You would need to download all of the .cha transcript files from TalkBank [Spanish](https://ca.talkbank.org/access/CallHome/spa.html).

Processing the transcripts takes place in multiple stages.

The first stage is to align the transcripts to the English translations, and to remove the .cha specific syntax.

The second is to apply the official PERL pre-processing scripts distributed as part of the Fisher/Callhome Spanish extension dataset containing English translations.

You can use the ```pipe.sh``` script to do this, but make sure to set the paths to the dataset and the relevant scripts first.
