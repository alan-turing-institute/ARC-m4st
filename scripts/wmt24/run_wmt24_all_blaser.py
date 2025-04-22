"""Runs BLASER over all sources from all categories."""

import argparse
import os

import pandas as pd

from m4st.metrics.base import TranslationDataset
from m4st.metrics.blaser import BLASERScore

map_lang = {
    "zh": "zho_Hans",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "en": "eng_Latn",
    "uk": "ukr_Cyrl",
    "is": "isl_Latn",
}


def main(args: dict) -> None:
    wmt_root = args["wmt_data_dir"]

    # Language pair to run
    lang_pair = args["lang_pair"]
    from_lang = lang_pair.split("-")[0]
    to_lang = lang_pair.split("-")[1]

    # Convert to SONAR language codes (from NLLB)
    from_lang_blaser = map_lang[from_lang]
    to_lang_blaser = map_lang[to_lang]

    # Read in sources
    sources_doc = f"{wmt_root}/sources/{lang_pair}.txt"
    with open(sources_doc) as input_file:
        sources = input_file.readlines()

    # Read in references
    refset = args["refset"]
    refs_doc = f"{wmt_root}/references/{lang_pair}.{refset}.txt"
    with open(refs_doc) as input_file:
        refs = input_file.readlines()

    # List of MT system results for this language pair
    mt_path = f"{wmt_root}/system-outputs/{lang_pair}"

    mt_names = []
    mt_results = []
    mt_translations = []
    mt_sources = []
    mt_refs = []
    mt_source_languages = []
    mt_ref_languages = []

    for mt_doc in os.scandir(mt_path):
        mt_name = os.path.basename(mt_doc).replace(".txt", "")
        mt_names.extend([mt_name] * len(sources))
        with open(mt_doc) as input_file:
            mt_translations.extend(input_file.readlines())
        mt_sources.extend(sources)
        mt_refs.extend(refs)
        mt_source_languages.extend([from_lang_blaser] * len(sources))
        mt_ref_languages.extend([to_lang_blaser] * len(sources))

    dataset = TranslationDataset(
        source=mt_sources,
        reference=mt_refs,
        prediction=mt_translations,
        source_language=mt_source_languages,
        target_language=mt_ref_languages,
    )

    # Output is stored as [output dir]/[xx-xx]/[MT system].[level].score
    output_dir = args["output_dir"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    lp_output_path = f"{output_dir}/{lang_pair}"
    os.makedirs(lp_output_path, exist_ok=True)

    # Set up and run BLASER
    blaser = BLASERScore(qe=False, audio_source=False)

    mt_results = blaser.get_scores(dataset)

    all_sys_results = pd.DataFrame({"system": mt_names, "score": mt_results})
    mt_output_file = f"{lp_output_path}/BLASERRef-{refset}.seg.score"
    all_sys_results.to_csv(mt_output_file, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refset",
        type=str,
        required=False,
        default="refA",
        help="Reference set to use. Must be either refA or refB.",
    )
    parser.add_argument(
        "--wmt-data-dir",
        type=str,
        required=True,
        help="Root directory for WMT data. This is the mt-metrics-eval-v2/wmt24 \
            directory as specified at  \
            https://github.com/google-research/mt-metrics-eval",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to hold outputs (BLASER scores). There will be one output csv \
            for each language pair.",
    )
    parser.add_argument(
        "--lang-pair",
        type=str,
        required=True,
        help="Language pair to run. Format should be xx-xx, e.g. en-de.",
    )

    args = parser.parse_args()
    main(vars(args))
