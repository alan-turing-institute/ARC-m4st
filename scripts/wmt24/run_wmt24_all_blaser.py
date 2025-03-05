"""Runs BLASER over all sources from all categories."""

import argparse
import os

import pandas as pd
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
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
    "uk": "uk_Cyrl",
    "is": "isl_Latn",
}


def main(args: dict) -> None:
    wmt_root = args["wmt_data_dir"]
    output_dir = args["output_dir"]
    refset = args["refset"]

    # Source text
    srcs_path = f"{wmt_root}/sources"

    # Reference text
    refs_path = f"{wmt_root}/references"

    # Translations
    mt_path = f"{wmt_root}/system-outputs"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Set up BLASER model and pipeline
    blaser_ref = load_blaser_model("blaser_2_0_ref").eval()

    text_encoder_model = load_sonar_text_encoder_model(
        "text_sonar_basic_encoder",
    ).eval()

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder=text_encoder_model, tokenizer="text_sonar_basic_encoder"
    )

    # Get list of directories, should be one for each langauge pair
    lp_dirs = os.scandir(mt_path)

    for lp_dir in lp_dirs:
        if os.path.isdir(lp_dir):
            # Get language pair str (xx-xx)
            lang_pair = os.path.basename(lp_dir).split("-")
            from_lang = lang_pair[0]  # Language translating from
            to_lang = lang_pair[1]  # Language translating to

            # Output is stored as [output dir]/[xx-xx]/[MT system].[level].score
            lp_output_path = f"{output_dir}/{from_lang}-{to_lang}"
            os.makedirs(lp_output_path, exist_ok=True)

            # Convert to SONAR language codes (from NLLB)
            from_lang_blaser = map_lang[from_lang]
            to_lang_blaser = map_lang[to_lang]

            # List of MT system results for this language pair
            mt_docs = os.scandir(lp_dir)
            # Doc containing list of sources for this language pair
            sources_doc = f"{srcs_path}/{from_lang}-{to_lang}.txt"
            # Doc containing list of references for this language pair
            refs_doc = f"{refs_path}/{from_lang}-{to_lang}.{refset}.txt"

            # Read in sources
            if os.path.exists(sources_doc):
                with open(sources_doc) as input_file:
                    sources = input_file.readlines()
            else:
                print(f"Could not find sources file: {sources_doc}")
                continue

            # Read in references
            if os.path.exists(refs_doc):
                with open(refs_doc) as input_file:
                    refs = input_file.readlines()
            else:
                print(f"Could not find reference file: {refs_doc}")
                continue

            # Get embeddings for sources
            src_embs = t2vec_model.predict(sources, source_lang=from_lang_blaser)

            # Get embeddings for references
            ref_embs = t2vec_model.predict(refs, source_lang=to_lang_blaser)

            mt_results = []
            mt_names = []

            for mt_doc in mt_docs:
                mt_name = os.path.basename(mt_doc).replace(".txt", "")
                mt_name_list = [mt_name] * len(src_embs)
                mt_names.extend(mt_name_list)
                mt_output_file = f"{lp_output_path}/{mt_name}-{refset}.seg.score"
                with open(mt_doc) as input_file:
                    translations = input_file.readlines()

                # Get embeddings for translations from this MT system
                t_embs = t2vec_model.predict(translations, source_lang=to_lang_blaser)
                for src, ref, mt in zip(src_embs, ref_embs, t_embs, strict=False):
                    result = blaser_ref(
                        src=src[None, :], ref=ref[None, :], mt=mt[None, :]
                    ).item()
                    mt_results.append(result)

            all_sys_results = pd.DataFrame({"system": mt_names, "score": mt_results})
            all_sys_results.to_csv(mt_output_file, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refset",
        type=str,
        required=False,
        default="RefA",
        help="Reference set to use. Must be either RefA or RefB.",
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

    args = parser.parse_args()
    main(vars(args))
