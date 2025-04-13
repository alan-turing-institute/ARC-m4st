import os

import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from tqdm import tqdm
from yaml import YAMLError, safe_load

from m4st.metrics import Metric, TranslationDataset


class BLASERScore(Metric):
    """Initialises and applies the BLASER 2.0 QE metric from the SONAR library."""

    def __init__(
        self,
        lang_code_config: str | os.PathLike,
        ref_lang_code: str = "eng_Latn",
        qe: bool = False,
    ) -> None:
        self.name = "blaser_2_0"
        self.data_req_inputs = ["prediction", "source", "source_language"]
        self.qe = qe
        if self.qe:
            self.name += "_qe"
        else:
            self.name += "_ref"
            self.data_req_inputs.append("reference")

        self.blaser = load_blaser_model(self.name).eval()
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

    def get_scores(self, dataset: TranslationDataset) -> list[float]:
        self.check_dataset_compatible(dataset)

        if self.lang_code_mapping:
            source_lang_codes = np.array(
                [self.lang_code_mapping[lng] for lng in dataset.source_language]
            )
        else:
            source_lang_codes = np.array(dataset.source_language)

        unique_languages = np.unique(source_lang_codes)

        # Store results for all languages so they can be returned together
        results = np.full(len(dataset), np.nan, dtype=float)

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for language in tqdm(unique_languages):
            mask = source_lang_codes == language
            sources_lang = np.array(dataset.source)[mask]
            preds_lang = np.array(dataset.prediction)[mask]

            # embed inputs
            embeds = {}
            embeds["src"] = self.text_embedder.predict(
                sources_lang, source_lang=language
            )
            embeds["mt"] = self.text_embedder.predict(
                preds_lang, source_lang=self.ref_lang_code
            )
            if not self.qe:
                refs_lang = np.array(dataset.reference)[mask]
                embeds["ref"] = self.text_embedder.predict(
                    refs_lang, source_lang=self.ref_lang_code
                )

            # dict of lists to list of dicts
            inputs = [
                {k: v[[i]] for k, v in embeds.items()}
                for i in range(len(embeds["src"]))
            ]

            lang_results = [self.blaser(**sample).item() for sample in tqdm(inputs)]
            results[mask] = lang_results

        return list(results)
