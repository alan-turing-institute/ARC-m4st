"""Runs BLASER over all sources in the "speech" category, which have to be matched
to the appropriate audio file.
"""

import argparse
import os

import pandas as pd
import torch
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from sonar.models.sonar_speech.loader import load_sonar_speech_model
from sonar.models.sonar_text import load_sonar_text_encoder_model

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

    # Set up BLASER model and pipelines
    # Need separate pipelines for generating speech and text embeddings
    blaser_ref = load_blaser_model("blaser_2_0_ref").eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lang_pair = os.path.basename(src_doc)[:5].split("-")
    from_lang = lang_pair[0]  # Language translating from
    to_lang = lang_pair[1]  # Language translating to

    if from_lang == "ja":
        speech_encoder = "sonar_speech_encoder_jpn"
    elif from_lang == "en":
        speech_encoder = "sonar_speech_encoder_eng"

    speech_encoder_model = load_sonar_speech_model(speech_encoder, device=device).eval()
    text_encoder_model = load_sonar_text_encoder_model(
        "text_sonar_basic_encoder", device=device
    ).eval()

    s2vec_model = SpeechToEmbeddingModelPipeline(encoder=speech_encoder_model)
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder=text_encoder_model, tokenizer="text_sonar_basic_encoder"
    )

    print(f"Processing sources from {src_doc}")

    au_src_results = []
    txt_src_results = []
    mt_sys_names = []

    output_file = os.path.join(output_dir, f"{from_lang}-{to_lang}_BLASER.csv")

    audio_subdir = f"test-{from_lang}-speech-audio"
    with open(src_doc) as input_file:
        srcs_list = input_file.readlines()

    from_lang_blaser = map_lang[from_lang]
    to_lang_blaser = map_lang[to_lang]

    # List of (line index, path) tuples for this translation pair
    speech_sources = [(i, s) for i, s in enumerate(srcs_list) if "speech" in s]

    audio_files = []
    src_texts = []
    ref_texts = []
    mt_texts = []

    for speech_src_file in speech_sources:
        # Get identifying filename for the corresponding .wav
        spch_identifier = speech_src_file[1].split("_", 1)[1].strip("\n")
        audio_file = f"{src_audio_dir}/{audio_subdir}/{spch_identifier}.wav"
        audio_files.append(audio_file)

        # Get corresponding source sentence
        src_sent_file = f"{srcs_path}/{from_lang}-{to_lang}.txt"
        with open(src_sent_file) as input_file:
            sentences = input_file.readlines()
            src_texts.append(sentences[speech_src_file[0]])

        # Get corresponding reference sentence
        ref_sent_file = f"{refs_path}/{from_lang}-{to_lang}.refA.txt"
        with open(ref_sent_file) as input_file:
            sentences = input_file.readlines()
            ref_texts.append(sentences[speech_src_file[0]])

    # Collect all machine translated sentences
    # For each translation system, select the line that matches the source
    mt_texts_sent = []
    mt_sent_dir = f"{mt_path}/{from_lang}-{to_lang}"
    mt_sent_files = os.scandir(mt_sent_dir)
    for mt_sent_file in mt_sent_files:
        mt_sys_names.append(os.path.basename(mt_sent_file.name))
        with open(mt_sent_file) as input_file:
            sentences = input_file.readlines()
            mt_texts_sent.append(sentences[speech_src_file[0]])

    mt_texts.append(mt_texts_sent)

    # Get embeddings for source text, references, and source audio
    # These will be common across translation models
    print("Getting audio source embeddings...")
    audio_src_embs = s2vec_model.predict(audio_files)
    print("Getting reference embeddings...")
    ref_embs = t2vec_model.predict(ref_texts, source_lang=to_lang_blaser)
    print("Getting text source embeddings...")
    src_embs = t2vec_model.predict(src_texts, source_lang=from_lang_blaser)

    print("Processing machine translations...")

    # There are multiple sets of machine translations
    # For each set of source, translation we apply the metric n times, once for each
    # translation model (n = ~25, seems to vary slightly by language)
    # mt_texts has shape (num_sentences, num_translation_models)
    for mt_set in mt_texts:  # For each sentence
        # Get embeddings for all translated versions of this sentence
        mt_embs = t2vec_model.predict(mt_set, source_lang=to_lang_blaser)

        # Compute metric for one (source, ref, translation) tuple at a time
        for au_emb, src_emb, ref_emb in zip(
            audio_src_embs, src_embs, ref_embs, strict=False
        ):
            for mt_emb in mt_embs:
                result_txt = blaser_ref(
                    src=src_emb[None, :], ref=ref_emb[None, :], mt=mt_emb[None, :]
                ).item()
                result_audio = blaser_ref(
                    src=au_emb[None, :], ref=ref_emb[None, :], mt=mt_emb[None, :]
                ).item()
                au_src_results.append(result_audio)
                txt_src_results.append(result_txt)

    results = pd.DataFrame(
        {
            "mt_system": mt_sys_names,
            "audio_source": au_src_results,
            "text_source": txt_src_results,
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
