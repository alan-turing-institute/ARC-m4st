import pandas as pd

from m4st.callhome.pipeline import CallhomePipeline
from m4st.metrics import (
    BLASERQEScore,
    BLASERRefScore,
    ChrFScore,
    COMETQEScore,
    COMETRefScore,
    SacreBLEUScore,
)
from m4st.translate.model import NLLBTranslateModel

if __name__ == "__main__":
    # Some test code
    text_folder = "/bask/projects/v/vjgo8416-spchmetrics/bv/ARC-m4st/data/\
spa_processed"
    audio_folder = None  # (
    #    "/Users/bvodenicharski/repos/ARC-m4st/experiments/callhome/data/eng/audio"
    # )
    eng_folder = "/bask/projects/v/vjgo8416-spchmetrics/bv/ARC-m4st/data/\
fisher_ch_spa-eng/data/corpus/ldc"
    mapping_folder = "/bask/projects/v/vjgo8416-spchmetrics/bv/ARC-m4st/data/\
fisher_ch_spa-eng/data/mapping"

    translation_model = NLLBTranslateModel("spa", "eng")
    transcription_model = None
    audio_snippet_dir = None  # "./audio_test_out"

    pipe = CallhomePipeline(
        text_folder,
        eng_folder,
        mapping_folder,
        audio_folder,
        translation_model,
        transcription_model,
        audio_snippet_dir=audio_snippet_dir,
    )

    metrics = [
        BLASERQEScore,
        BLASERRefScore,
        ChrFScore,
        COMETQEScore,
        COMETRefScore,
        SacreBLEUScore,
    ]

    originals = []
    references = []
    translations = []
    # TODO NLLB sometimes does not translate the text at all :(
    for text_dict in iter(pipe):
        originals.append(text_dict["original"])
        references.append(text_dict["reference_english"])
        translations.append(text_dict["translated_text"])

    originals = pd.Series(originals)
    references = pd.Series(references)
    translations = pd.Series(translations)
    src_lang_codes = pd.Series(["spa_Latn" for _ in range(len(originals))])

    all_metric_evals = {}
    for metric_class in metrics:
        metric = metric_class()
        if metric_class.__name__ in ["COMETRefScore", "COMETQEScore"]:
            metric_evals = metric.get_scores(references, translations, originals)
        elif metric_class.__name__ in ["BLASERQEScore"]:
            metric_evals = metric.get_scores(translations, originals, src_lang_codes)
        elif metric_class.__name__ in ["BLASERRefScore"]:
            metric_evals = metric.get_scores(
                references, translations, originals, src_lang_codes
            )
        else:
            metric_evals = metric.get_scores(references, translations)

        all_metric_evals[metric_class.__name__] = metric_evals

    # Add the text, so we can tell which dialogue gave rise to the evals:
    all_metric_evals["originals"] = originals
    all_metric_evals["references"] = references
    all_metric_evals["translations"] = translations

    df = pd.DataFrame.from_dict(all_metric_evals)
    df.to_json("correlations.json", index=False)
