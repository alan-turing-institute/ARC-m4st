"""Runs BLASER over all sources in the "speech" category, which have to be matched
to the appropriate audio file.
"""

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
    src_audio_dir = args["source_audio_dir"]
    lang_pair = args["lang_pair"]

    print("Setting up paths...")

    # Source text
    srcs_path = f"{wmt_root}/sources"

    # Reference text
    refs_path = f"{wmt_root}/references"

    # Source mappings
    src_doc = f"{wmt_root}/documents/{lang_pair}.docs"

    # Translations (English-German)
    mt_path = f"{wmt_root}/system-outputs"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    lang_pair = os.path.basename(src_doc)[:5].split("-")
    from_lang = lang_pair[0]  # Language translating from
    to_lang = lang_pair[1]  # Language translating to

    print(f"Processing sources from {src_doc}")
    output_file = os.path.join(output_dir, f"{from_lang}-{to_lang}_BLASER.csv")

    audio_subdir = f"test-{from_lang}-speech-audio-resampled"
    with open(src_doc) as input_file:
        srcs_list = input_file.readlines()

    from_lang_blaser = map_lang[from_lang]
    to_lang_blaser = map_lang[to_lang]

    # List of (line index, path) tuples for this translation pair
    speech_sources = [(i, s) for i, s in enumerate(srcs_list) if "speech" in s]

    mt_sys_names = []
    audio_files = []
    src_texts = []
    ref_texts = []
    mt_texts = []

    # For all sentences in this translation pair set in the speech domain
    for speech_src_details in speech_sources:
        # Get identifying filename for the corresponding .wav
        spch_identifier = speech_src_details[1].split("_", 1)[1].strip("\n")
        audio_file = f"{src_audio_dir}/{audio_subdir}/{spch_identifier}.wav"
        audio_files.append(audio_file)

        # Get corresponding source sentence
        src_sent_file = f"{srcs_path}/{from_lang}-{to_lang}.txt"
        with open(src_sent_file) as input_file:
            sentences = input_file.readlines()
            src_texts.append(sentences[speech_src_details[0]])

        # Get corresponding reference sentence
        ref_sent_file = f"{refs_path}/{from_lang}-{to_lang}.refA.txt"
        with open(ref_sent_file) as input_file:
            sentences = input_file.readlines()
            ref_texts.append(sentences[speech_src_details[0]])

        # Collect all machine translated sentences
        # For each translation system, select the line that matches the source
        mt_texts_sent = []
        mt_system_sent = []
        mt_sent_dir = f"{mt_path}/{from_lang}-{to_lang}"

        # For each file containing the results from one MT system
        for mt_sent_file in os.scandir(mt_sent_dir):  # ~25
            mt_name = os.path.basename(mt_sent_file).replace(".txt", "")
            mt_system_sent.append(mt_name)
            with open(mt_sent_file) as input_file:
                sentences = input_file.readlines()
                mt_texts_sent.append(sentences[speech_src_details[0]])

        mt_texts.append(mt_texts_sent)
        mt_sys_names.extend(mt_system_sent)

    print(len(mt_texts))  # 111, 25 (num_sentences, num_translation_models)
    print(len(mt_sys_names))  # 2331 (num_sentences * num_translation models)

    print("Processing machine translations...")

    # au, src, ref should all have length 111
    print(len(audio_files), len(src_texts), len(ref_texts))

    # There are multiple sets of machine translations
    # For each set of source, translation we apply the metric n times, once for each
    # translation model (n = ~25, seems to vary slightly by language)
    # mt_texts has shape (num_sentences, num_translation_models)
    result_audio_src = []
    result_txt_src = []
    for mt_set in mt_texts:
        # Run BLASER on audio sources
        blaser = BLASERScore(qe=False, audio_source=True)
        result_audio_src.append(
            blaser.get_scores(
                TranslationDataset(
                    source=audio_files,
                    reference=ref_texts,
                    prediction=mt_set,
                    source_language=[from_lang_blaser] * len(audio_files),
                    target_language=[to_lang_blaser] * len(audio_files),
                )
            )
        )

        # Run BLASER on text sources
        blaser = BLASERScore(qe=False, audio_source=False)
        result_txt_src.append(
            blaser.get_scores(
                TranslationDataset(
                    source=src_texts,
                    reference=ref_texts,
                    prediction=mt_set,
                    source_language=[from_lang_blaser] * len(src_texts),
                    target_language=[to_lang_blaser] * len(src_texts),
                )
            )
        )

    results = pd.DataFrame(
        {
            "mt_system": mt_sys_names,
            "audio_source": result_audio_src,
            "text_source": result_txt_src,
        }
    )
    results.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-audio-dir",
        type=str,
        required=True,
        help="Directory holding WAV files for WMT24 speech test set.",
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
        "--lang-pair",
        type=str,
        required=True,
        help="Language pair to run. Format should be xx-xx, e.g. en-de.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to hold outputs (BLASER scores). There will be one output csv \
            for each language pair.",
    )

    args = parser.parse_args()
    main(vars(args))
