import json
import os
import tempfile
from abc import ABC, abstractmethod

import evaluate
import numpy as np
from datasets import Dataset
from metricx import MT5ForRegression, get_dataset
from pandas import DataFrame

# from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
# from sonar.models.blaser.loader import load_blaser_model
from transformers import AutoTokenizer, Trainer, TrainingArguments
from yaml import YAMLError, safe_load


class Metric(ABC):
    @abstractmethod
    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        """Function definition for all metrics. Assumes use of the DEMETR dataset.

        cat_data --     pd.DataFrame object created by directly loading a DEMETR JSON
                        file (e.g. base_id35_reference.json) with pd.read_json.
        output_path --  Directory for storing output JSON files. There will be one
                        output file for each DEMETR category, for each metric.
        input_fp --     Path to input JSON file from the DEMETR dataset.
        ghfghgfhj
        """


class ChrFScore(Metric):
    """Applies ChrF/++ from the evaluate library.
    When word_order=0 (default) computes original ChrF metric without including word
    n-grams. When word_order=2, computes ChrF++. The DEMETR paper refers to ChrF++
    as ChrF2.For more details see https://huggingface.co/spaces/evaluate-metric/chrf"""

    def __init__(self, word_order: int = 0) -> None:
        self.chrf = evaluate.load("chrf")
        self.word_order = word_order

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"ChrF{self.word_order}_{input_fp}"
        results = {}
        # ID, language, mt_score, perturbed_score
        ref_txts = cat_data["eng_sent"]  # Human translation
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_langs = cat_data["lang_tag"]  # Source language

        for index, ref_txt in ref_txts.items():
            mt_txt = mt_txts[index]
            d_txt = dfluent_txts[index]
            lang = src_langs[index]
            mt_score = self.chrf.compute(
                predictions=[mt_txt],
                references=[[ref_txt]],
                word_order=self.word_order,
            )
            d_score = self.chrf.compute(
                predictions=[d_txt],
                references=[[ref_txt]],
                word_order=self.word_order,
            )
            results[int(cat_data["id"][index])] = {
                "source_language": lang,
                "mt_score": mt_score["score"],
                "disfluent_score": d_score["score"],
            }
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)


class BLEUScore(Metric):
    """Applies SacreBleu from the evaluate library."""

    def __init__(self) -> None:
        self.bleu = evaluate.load("sacrebleu")

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"BLEU_{input_fp}"
        ref_txts = cat_data["eng_sent"]  # Human translation
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_langs = cat_data["lang_tag"]  # Source language

        results = {}

        # SacreBleu doesn't seem to support batching that isn't document-level, so
        # each sentence must be run through separately
        for index, ref_txt in ref_txts.items():
            mt_txt = mt_txts[index]
            d_txt = dfluent_txts[index]
            lang = src_langs[index]
            mt_score = self.bleu.compute(predictions=[mt_txt], references=[[ref_txt]])
            d_score = self.bleu.compute(predictions=[d_txt], references=[[ref_txt]])

            results[int(cat_data["id"][index])] = {
                "source_language": lang,
                "mt_score": mt_score["score"],
                "disfluent_score": d_score["score"],
            }
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)


# class BLASERRefScore(Metric):
#     """Initialises and applies the BLASER 2.0 QE metric from the SONAR library."""

#     def __init__(
#         self,
#         lang_code_config: str | os.PathLike,
#         ref_lang_code: str = "eng_Latn",
#     ) -> None:
#         self.blaser_ref = load_blaser_model("blaser_2_0_ref").eval()
#         self.text_embedder = TextToEmbeddingModelPipeline(
#             encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
#         )
#         # Code defining the target language
#         # Defaults to English
#         self.ref_lang_code = ref_lang_code

#         # Source language code must be provided to generate SONAR embeddings
#         # If a config is provided, it will be used to map language codes in the
#         # dataset to SONAR-recognised codes
#         if lang_code_config:
#             with open(lang_code_config) as stream:
#                 try:
#                     lang_code_mapping = safe_load(stream)
#                 except YAMLError as exc:
#                     print(exc)
#             self.lang_code_mapping = lang_code_mapping

#     def get_scores(
#         self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
#     ) -> None:
#         output_file = f"BLASER_Ref_{input_fp}"
#         ref_txts = cat_data["eng_sent"]  # Human translation
#         mt_txts = cat_data["mt_sent"]  # Original machine translation
#         dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
#         src_txts = cat_data["src_sent"]  # Source (original) text
#         src_langs = cat_data["lang_tag"]  # Source language
#         sentence_ids = cat_data["id"]
#         if self.lang_code_mapping:
#             source_lang_codes = src_langs.replace(self.lang_code_mapping)
#         else:
#             source_lang_codes = src_langs
#         langs = np.unique(source_lang_codes)

#         results = {}

#         # BLASER requires the source language, so at best we can batch embedding
#         # generation by language as source_lang must be a string
#         for language in langs:
#             mask = source_lang_codes == language
#             sources_lang = np.array(src_txts[mask])
#             refs_lang = np.array(ref_txts[mask])
#             mt_lang = np.array(mt_txts[mask])
#             d_lang = np.array(dfluent_txts[mask])
#             src_lang = np.array(src_langs[mask])
#             ids_lang = np.array(sentence_ids[mask])

#             # Source embeddings
#             src_embs = self.text_embedder.predict(sources_lang, source_lang=language)

#             # Reference embeddings
#             ref_embs = self.text_embedder.predict(
#                 refs_lang, source_lang=self.ref_lang_code
#             )
#             # Fluent translation embeddings
#             mt_embs = self.text_embedder.predict(
#                 mt_lang, source_lang=self.ref_lang_code
#             )
#             # Disfluent translation embeddings
#             d_embs = self.text_embedder.predict(d_lang, source_lang=self.ref_lang_code)

#             # Actual metric is computed one sample at a time
#             for i in range(len(src_embs)):
#                 mt_result = self.blaser_ref(
#                     src=src_embs[[i]], ref=ref_embs[[i]], mt=mt_embs[[i]]
#                 ).item()
#                 d_result = self.blaser_ref(
#                     src=src_embs[[i]], ref=ref_embs[[i]], mt=d_embs[[i]]
#                 ).item()

#                 results[int(ids_lang[[i]])] = {
#                     "source_language": src_lang[i],
#                     "mt_score": mt_result,
#                     "disfluent_score": d_result,
#                 }

#         with open(os.path.join(output_path, output_file), "w+") as file_to_write:
#             json.dump(results, file_to_write)


# class BLASERQEScore(Metric):
#     """Initialises and applies the BLASER 2.0 reference-based metric from the SONAR
#     library."""

#     def __init__(
#         self, lang_code_config: str | os.PathLike, ref_lang_code: str = "eng_Latn"
#     ) -> None:
#         self.blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
#         self.text_embedder = TextToEmbeddingModelPipeline(
#             encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
#         )
#         # Code defining the target language
#         # Defaults to English
#         self.ref_lang_code = ref_lang_code

#         # Source language code must be provided to generate SONAR embeddings
#         # If a config is provided, it will be used to map language codes in the
#         # dataset to SONAR-recognised codes
#         if lang_code_config:
#             with open(lang_code_config) as stream:
#                 try:
#                     lang_code_mapping = safe_load(stream)
#                 except YAMLError as exc:
#                     print(exc)
#             self.lang_code_mapping = lang_code_mapping

#     def get_scores(
#         self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
#     ) -> None:
#         output_file = f"BLASER_QE_{input_fp}"
#         mt_txts = cat_data["mt_sent"]  # Original machine translation
#         dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
#         src_txts = cat_data["src_sent"]  # Source (original) text
#         src_langs = cat_data["lang_tag"]  # Source language
#         sentence_ids = cat_data["id"]
#         if self.lang_code_mapping:
#             source_lang_codes = src_langs.replace(self.lang_code_mapping)
#         else:
#             source_lang_codes = src_langs
#         langs = np.unique(source_lang_codes)

#         results = {}

#         # BLASER requires the source language, so at best we can batch by language as
#         # source_lang must be a string
#         for language in langs:
#             mask = source_lang_codes == language
#             sources_lang = np.array(src_txts[mask])
#             mt_lang = np.array(mt_txts[mask])
#             d_lang = np.array(dfluent_txts[mask])
#             src_lang = np.array(src_langs[mask])
#             ids_lang = np.array(sentence_ids[mask])

#             # Source embeddings
#             src_embs = self.text_embedder.predict(sources_lang, source_lang=language)

#             # Fluent translation embeddings
#             mt_embs = self.text_embedder.predict(
#                 mt_lang, source_lang=self.ref_lang_code
#             )

#             # Disfluent translation embeddings
#             d_embs = self.text_embedder.predict(d_lang, source_lang=self.ref_lang_code)

#             for i in range(len(src_embs)):
#                 mt_result = self.blaser_qe(src=src_embs[[i]], mt=mt_embs[[i]]).item()
#                 d_result = self.blaser_qe(src=src_embs[[i]], mt=d_embs[[i]]).item()

#                 results[int(ids_lang[[i]])] = {
#                     "source_language": src_lang[i],
#                     "mt_score": mt_result,
#                     "disfluent_score": d_result,
#                 }

#         with open(os.path.join(output_path, output_file), "w+") as file_to_write:
#             json.dump(results, file_to_write)


class COMETScore(Metric):
    """Applies COMET metric using the evaluate library.
    All COMET models require the same input footprint, including QE versions. You can
    switch between different COMET models by providing the model argument.
    e.g. model="wmt21-comet-qe-mqmq", model="Unbabel/XCOMET-XXL".
    See https://huggingface.co/spaces/evaluate-metric/comet for more details.
    """

    def __init__(self, model: str = "wmt21-comet-mqm") -> None:
        self.model = model
        self.comet = evaluate.load("comet", self.model)

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"{self.model}_{input_fp}"
        sentence_ids = np.array(cat_data["id"])
        src_langs = list(cat_data["lang_tag"])

        mt_scores = self.comet.compute(
            predictions=cat_data["mt_sent"],
            references=cat_data["eng_sent"],
            sources=cat_data["src_sent"],
        )
        d_scores = self.comet.compute(
            predictions=cat_data["pert_sent"],
            references=cat_data["eng_sent"],
            sources=cat_data["src_sent"],
        )

        results = {}

        for i in range(len(mt_scores["scores"])):
            results[int(sentence_ids[[i]])] = {
                "source_language": src_langs[i],
                "mt_score": mt_scores["scores"][i],
                "disfluent_score": d_scores["scores"][i],
            }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)


class MetricXScore(Metric):
    """Applies MetricX 2024: https://github.com/google-research/metricx"""

    def __init__(
        self,
        model: str = "google/metricx-24-hybrid-xl-v2p6",
        tokenizer: str = "google/mt5-xl",
        max_input_length: int = 1536,
        batch_size: int = 1,
        qe: bool = False,
    ) -> None:
        self.model_name = model.replace("/", "_")  # for output file paths
        self.model = MT5ForRegression.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.qe = qe

    def preprocess(
        self, cat_data: DataFrame, src_col: str, pred_col: str, ref_col: str
    ) -> Dataset:
        inputs = [
            {
                "source": row[src_col],
                "hypothesis": row[pred_col],
                "reference": row[ref_col],
            }
            for _, row in cat_data.iterrows()
        ]
        with tempfile.NamedTemporaryFile() as fp:
            json.dump(inputs, fp)
            return get_dataset(fp.name, self.tokenizer, self.max_input_length, self.qe)

    def compute(self, ds: Dataset) -> list[float]:
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir, per_device_eval_batch_size=self.batch_size
            )
            trainer = Trainer(model=self.model, args=training_args)
            predictions, _, _ = trainer.predict(test_dataset=ds["test"])

        return [float(pred) for pred in predictions]

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"{self.model_name}_{input_fp}"
        sentence_ids = np.array(cat_data["id"])
        src_langs = list(cat_data["lang_tag"])

        mt_data = self.preprocess(
            cat_data, src_col="src_sent", pred_col="mt_sent", ref_col="eng_sent"
        )
        mt_scores = self.compute(mt_data)

        d_data = self.preprocess(
            cat_data, src_col="src_sent", pred_col="pert_sent", ref_col="eng_sent"
        )
        d_scores = self.compute(d_data)

        results = {}

        for i in range(len(mt_scores["scores"])):
            results[int(sentence_ids[[i]])] = {
                "source_language": src_langs[i],
                "mt_score": mt_scores[i],
                "disfluent_score": d_scores[i],
            }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
