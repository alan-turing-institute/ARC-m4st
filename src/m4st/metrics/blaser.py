import json
import os

import numpy as np
from pandas import DataFrame
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from yaml import YAMLError, safe_load

from m4st.metrics import Metric


class BLASERRefScore(Metric):
    """Initialises and applies the BLASER 2.0 QE metric from the SONAR library."""

    def __init__(
        self,
        lang_code_config: str | os.PathLike,
        ref_lang_code: str = "eng_Latn",
    ) -> None:
        self.blaser_ref = load_blaser_model("blaser_2_0_ref").eval()
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )
        # Code defining the target language
        # Defaults to English
        self.ref_lang_code = ref_lang_code

        # Source language code must be provided to generate SONAR embeddings
        # If a config is provided, it will be used to map language codes in the
        # dataset to SONAR-recognised codes
        if lang_code_config:
            with open(lang_code_config) as stream:
                try:
                    lang_code_mapping = safe_load(stream)
                except YAMLError as exc:
                    print(exc)
            self.lang_code_mapping = lang_code_mapping

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"BLASER_Ref_{input_fp}"
        ref_txts = cat_data["eng_sent"]  # Human translation
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_txts = cat_data["src_sent"]  # Source (original) text
        src_langs = cat_data["lang_tag"]  # Source language
        sentence_ids = cat_data["id"]
        if self.lang_code_mapping:
            source_lang_codes = src_langs.replace(self.lang_code_mapping)
        else:
            source_lang_codes = src_langs
        langs = np.unique(source_lang_codes)

        results = {}

        # BLASER requires the source language, so at best we can batch embedding
        # generation by language as source_lang must be a string
        for language in langs:
            mask = source_lang_codes == language
            sources_lang = np.array(src_txts[mask])
            refs_lang = np.array(ref_txts[mask])
            mt_lang = np.array(mt_txts[mask])
            d_lang = np.array(dfluent_txts[mask])
            src_lang = np.array(src_langs[mask])
            ids_lang = np.array(sentence_ids[mask])

            # Source embeddings
            src_embs = self.text_embedder.predict(sources_lang, source_lang=language)

            # Reference embeddings
            ref_embs = self.text_embedder.predict(
                refs_lang, source_lang=self.ref_lang_code
            )
            # Fluent translation embeddings
            mt_embs = self.text_embedder.predict(
                mt_lang, source_lang=self.ref_lang_code
            )
            # Disfluent translation embeddings
            d_embs = self.text_embedder.predict(d_lang, source_lang=self.ref_lang_code)

            # Actual metric is computed one sample at a time
            for i in range(len(src_embs)):
                mt_result = self.blaser_ref(
                    src=src_embs[[i]], ref=ref_embs[[i]], mt=mt_embs[[i]]
                ).item()
                d_result = self.blaser_ref(
                    src=src_embs[[i]], ref=ref_embs[[i]], mt=d_embs[[i]]
                ).item()

                results[int(ids_lang[[i]])] = {
                    "source_language": src_lang[i],
                    "mt_score": mt_result,
                    "disfluent_score": d_result,
                }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)


class BLASERQEScore(Metric):
    """Initialises and applies the BLASER 2.0 reference-based metric from the SONAR
    library."""

    def __init__(
        self, lang_code_config: str | os.PathLike, ref_lang_code: str = "eng_Latn"
    ) -> None:
        self.blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )
        # Code defining the target language
        # Defaults to English
        self.ref_lang_code = ref_lang_code

        # Source language code must be provided to generate SONAR embeddings
        # If a config is provided, it will be used to map language codes in the
        # dataset to SONAR-recognised codes
        if lang_code_config:
            with open(lang_code_config) as stream:
                try:
                    lang_code_mapping = safe_load(stream)
                except YAMLError as exc:
                    print(exc)
            self.lang_code_mapping = lang_code_mapping

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"BLASER_QE_{input_fp}"
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_txts = cat_data["src_sent"]  # Source (original) text
        src_langs = cat_data["lang_tag"]  # Source language
        sentence_ids = cat_data["id"]
        if self.lang_code_mapping:
            source_lang_codes = src_langs.replace(self.lang_code_mapping)
        else:
            source_lang_codes = src_langs
        langs = np.unique(source_lang_codes)

        results = {}

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for language in langs:
            mask = source_lang_codes == language
            sources_lang = np.array(src_txts[mask])
            mt_lang = np.array(mt_txts[mask])
            d_lang = np.array(dfluent_txts[mask])
            src_lang = np.array(src_langs[mask])
            ids_lang = np.array(sentence_ids[mask])

            # Source embeddings
            src_embs = self.text_embedder.predict(sources_lang, source_lang=language)

            # Fluent translation embeddings
            mt_embs = self.text_embedder.predict(
                mt_lang, source_lang=self.ref_lang_code
            )

            # Disfluent translation embeddings
            d_embs = self.text_embedder.predict(d_lang, source_lang=self.ref_lang_code)

            for i in range(len(src_embs)):
                mt_result = self.blaser_qe(src=src_embs[[i]], mt=mt_embs[[i]]).item()
                d_result = self.blaser_qe(src=src_embs[[i]], mt=d_embs[[i]]).item()

                results[int(ids_lang[[i]])] = {
                    "source_language": src_lang[i],
                    "mt_score": mt_result,
                    "disfluent_score": d_result,
                }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
