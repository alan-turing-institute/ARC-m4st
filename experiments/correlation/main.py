import argparse

import pandas as pd

from m4st.callhome.pipeline import CallhomePipeline
from m4st.metrics import TranslationDataset
from m4st.metrics.blaser import BLASERScore
from m4st.metrics.comet import COMETScore
from m4st.metrics.string import BLEUScore, ChrFScore
from m4st.translate.model import NLLBTranslateModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Requires the processed CallHome + Fisher Spa-Eng dataset. \
Refer to the repo README for how to process, or ask Jack."
    )

    parser.add_argument(
        "--text_folder",
        type=str,
        required=True,
        help="Path to the folder containing processed text files. Refer \
to the CallHome processing README for how to process the dataset.",
    )

    parser.add_argument(
        "--audio_folder",
        type=str,
        default=None,
        help="Path to the folder containing audio files. Optional. The metrics ",
    )

    parser.add_argument(
        "--eng_folder",
        type=str,
        required=True,
        help="Path to the folder containing English translation file. \
This requires the paid version of the Fisher Spa-Eng corput, and can be \
found under fisher_ch_spa-eng/data/corpus/ldc.",
    )

    parser.add_argument(
        "--mapping_folder",
        type=str,
        required=True,
        help="Path to the folder containing mapping file. Found under \
fisher_ch_spa-eng/data/mapping for the Fisher dataset.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text_folder = args.text_folder
    audio_folder = args.audio_folder
    eng_folder = args.eng_folder
    mapping_folder = args.mapping_folder

    translation_model = NLLBTranslateModel("spa", "eng")
    # Originally intended to do an experiment with transcription. Out of time.
    transcription_model = None
    # If this is set, the pipeline will extract the audio. Otherwise skip the step.
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

    # Keeping the names the same as in report; hardcoded in downstream scripts.
    metrics = {
        "BLASERQEScore": lambda: BLASERScore(qe=True),
        "BLASERRefScore": lambda: BLASERScore(qe=False),
        "ChrFScore": lambda: ChrFScore(),
        "COMETQEScore": lambda: COMETScore(),
        "SacreBLEUScore": lambda: BLEUScore(),
    }

    originals = []
    references = []
    translations = []
    for text_dict in iter(pipe):
        originals.append(text_dict["original"])
        references.append(text_dict["reference_english"])
        translations.append(text_dict["translated_text"])
        break  # TODO Rm, for testing...

    originals = pd.Series(originals)
    references = pd.Series(references)
    translations = pd.Series(translations)
    src_lang_codes = pd.Series(["spa_Latn" for _ in range(len(originals))])
    tar_lang_codes = pd.Series(["eng_Latn" for _ in range(len(originals))])

    all_metric_evals = {}
    for metric_name, metric_init_fn in metrics.items():
        metric = metric_init_fn()

        metric_evals = metric.get_scores(
            TranslationDataset(
                source=originals,
                prediction=translations,
                reference=references,
                source_language=src_lang_codes,
                target_language=tar_lang_codes,
            )
        )

        all_metric_evals[metric_name] = metric_evals

    # Add the text, so we can tell which dialogue gave rise to the evals:
    all_metric_evals["originals"] = originals
    all_metric_evals["references"] = references
    all_metric_evals["translations"] = translations

    df = pd.DataFrame.from_dict(all_metric_evals)
    df.to_json("metric_evaluation.json", index=False)
