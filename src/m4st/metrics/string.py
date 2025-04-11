import json
import os

import evaluate
from pandas import DataFrame, Series

from m4st.metrics import Metric


class ChrFScore:
    """Applies ChrF/++ from the evaluate library.
    When word_order=0 (default) computes original ChrF metric without including word
    n-grams. When word_order=2, computes ChrF++. The DEMETR paper refers to ChrF++
    as ChrF2.For more details see https://huggingface.co/spaces/evaluate-metric/chrf"""

    def __init__(self, word_order: int = 0) -> None:
        self.chrf = evaluate.load("chrf")
        self.word_order = word_order

    def get_scores(self, references: Series, predictions: Series) -> list:
        return [
            self.chrf.compute(
                predictions=[mt_txt],
                references=[[ref_txt]],
                word_order=self.word_order,
                eps_smoothing=True,
            )["score"]
            for ref_txt, mt_txt in zip(references, predictions, strict=True)
        ]

    def process_demetr_cat(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        # ID, language, mt_score, perturbed_score
        ref_txts = cat_data["eng_sent"]  # Human translation
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_langs = cat_data["lang_tag"]  # Source language
        sentence_ids = cat_data["id"]  # Perturbation category IDs

        mt_scores = self.get_scores(ref_txts, mt_txts)
        d_scores = self.get_scores(ref_txts, dfluent_txts)

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
        output_file = f"ChrF{self.word_order}_{input_fp}"
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)


class BLEUScore(Metric):
    """Applies SacreBleu from the evaluate library."""

    def __init__(self) -> None:
        self.bleu = evaluate.load("sacrebleu")

    def get_scores(self, references: Series, predictions: Series) -> list:
        return [
            self.bleu.compute(predictions=[mt_txt], references=[[ref_txt]])["score"]
            for ref_txt, mt_txt in zip(references, predictions, strict=True)
        ]

    def process_demetr_cat(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        # ID, language, mt_score, perturbed_score
        ref_txts = cat_data["eng_sent"]  # Human translation
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_langs = cat_data["lang_tag"]  # Source language
        cat_ids = cat_data["id"]  # Perturbation category IDs

        mt_scores = self.get_scores(ref_txts, mt_txts)
        d_scores = self.get_scores(ref_txts, dfluent_txts)

        results = {
            index: {
                "source_language": lang,
                "mt_score": mt_score,
                "disfluent_score": d_score,
            }
            for index, lang, mt_score, d_score in zip(
                cat_ids, src_langs, mt_scores, d_scores, strict=True
            )
        }
        output_file = f"BLEU_{input_fp}"
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
