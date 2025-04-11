import json
import os

import numpy as np
from pandas import DataFrame, Series
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
        self,
        references: Series,
        predictions: Series,
        sources: Series,
        source_lang_codes: Series,
    ) -> list:
        langs = np.unique(source_lang_codes)

        # Store results for all languages so they can be returned together
        results = np.full(len(references), np.nan, dtype=float)

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for language in langs:
            mask = source_lang_codes == language
            sources_lang = np.array(sources[mask])
            refs_lang = np.array(references[mask])
            preds_lang = np.array(predictions[mask])

            src_embs = self.text_embedder.predict(sources_lang, source_lang=language)
            ref_embs = self.text_embedder.predict(
                refs_lang, source_lang=self.ref_lang_code
            )
            mt_embs = self.text_embedder.predict(
                preds_lang, source_lang=self.ref_lang_code
            )

            lang_results = [
                self.blaser_ref(
                    src=src_embs[[i]], ref=ref_embs[[i]], mt=mt_embs[[i]]
                ).item()
                for i in range(len(src_embs))
            ]
            results[mask] = lang_results

        return results

    def process_demetr_cat(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
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

        mt_scores = self.get_scores(ref_txts, mt_txts, src_txts, source_lang_codes)
        d_scores = self.get_scores(ref_txts, dfluent_txts, src_txts, source_lang_codes)

        results = {
            index: {
                "source_language": lang,
                "mt_score": mt_score,
                "disfluent_score": d_score,
            }
            for index, lang, mt_score, d_score in zip(
                sentence_ids, src_langs, mt_scores, d_scores, strict=True
            )
        }
        output_file = f"BLASER_Ref_{input_fp}"
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)


class BLASERQEScore(Metric):
    """Initialises and applies the BLASER 2.0 QE metric from the SONAR library."""

    def __init__(
        self,
        lang_code_config: str | os.PathLike,
        ref_lang_code: str = "eng_Latn",
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
        self,
        predictions: Series,
        sources: Series,
        source_lang_codes: Series,
    ) -> list:
        langs = np.unique(source_lang_codes)

        # Store results for all languages so they can be returned together
        results = np.full(len(sources), np.nan, dtype=float)

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for language in langs:
            mask = source_lang_codes == language
            sources_lang = np.array(sources[mask])
            preds_lang = np.array(predictions[mask])

            src_embs = self.text_embedder.predict(sources_lang, source_lang=language)
            mt_embs = self.text_embedder.predict(
                preds_lang, source_lang=self.ref_lang_code
            )

            lang_results = [
                self.blaser_qe(src=src_embs[[i]], mt=mt_embs[[i]]).item()
                for i in range(len(src_embs))
            ]
            results[mask] = lang_results

        return results

    def process_demetr_cat(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_txts = cat_data["src_sent"]  # Source (original) text
        src_langs = cat_data["lang_tag"]  # Source language
        sentence_ids = cat_data["id"]
        if self.lang_code_mapping:
            source_lang_codes = src_langs.replace(self.lang_code_mapping)
        else:
            source_lang_codes = src_langs

        mt_scores = self.get_scores(mt_txts, src_txts, source_lang_codes)
        d_scores = self.get_scores(dfluent_txts, src_txts, source_lang_codes)

        results = {
            index: {
                "source_language": lang,
                "mt_score": mt_score,
                "disfluent_score": d_score,
            }
            for index, lang, mt_score, d_score in zip(
                sentence_ids, src_langs, mt_scores, d_scores, strict=True
            )
        }
        output_file = f"BLASER_QE_{input_fp}"
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
