import json
import os

import evaluate
from pandas import DataFrame

from m4st.metrics import Metric


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
