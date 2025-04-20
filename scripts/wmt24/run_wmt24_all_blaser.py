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
    output_dir = args["output_dir"]
    refset = args["refset"]
    lang_pair = args["lang_pair"]

    # Source text
    sources_doc = f"{wmt_root}/sources/{lang_pair}.txt"

    # Reference text
    refs_doc = f"{wmt_root}/references/{lang_pair}.{refset}.txt"

    # Translations
    mt_path = f"{wmt_root}/system-outputs/{lang_pair}"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Set up BLASER
    blaser = BLASERScore(qe=False, audio_source=False)

    if os.path.isdir(mt_path):
        # Output is stored as [output dir]/[xx-xx]/[MT system].[level].score
        lp_output_path = f"{output_dir}/{lang_pair}"
        os.makedirs(lp_output_path, exist_ok=True)

        from_lang = lang_pair.split("-")[0]
        to_lang = lang_pair.split("-")[1]

        # Convert to SONAR language codes (from NLLB)
        from_lang_blaser = map_lang[from_lang]
        to_lang_blaser = map_lang[to_lang]

        # Read in sources
        if os.path.exists(sources_doc):
            with open(sources_doc) as input_file:
                sources = input_file.readlines()

            # Read in references
            if os.path.exists(refs_doc):
                with open(refs_doc) as input_file:
                    refs = input_file.readlines()

                mt_names = []

                # List of MT system results for this language pair
                for mt_doc in os.scandir(mt_path):
                    mt_name = os.path.basename(mt_doc).replace(".txt", "")
                    mt_name_list = [mt_name] * len(sources)
                    mt_names.extend(mt_name_list)
                    mt_output_file = f"{lp_output_path}/BLASERRef-{refset}.seg.score"
                    with open(mt_doc) as input_file:
                        translations = input_file.readlines()

                    dataset = TranslationDataset(
                        source=sources,
                        reference=refs,
                        prediction=translations,
                        source_language=[from_lang_blaser] * len(sources),
                        target_language=[to_lang_blaser] * len(sources),
                    )

                    mt_results = blaser.get_scores(dataset)

                all_sys_results = pd.DataFrame(
                    {"system": mt_names, "score": mt_results}
                )
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
