"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import csv
import os

import numpy as np
import pandas as pd

from m4st.metrics import (
    BLASERQEScore,
    BLASERRefScore,
    ChrFScore,
    COMETQEScore,
    COMETRefScore,
    SacreBLEUScore,
)


class ProcessDEMETR:
    def __init__(
        self,
        output_filepath: os.PathLike | str,
        demetr_root: os.PathLike | str,
        metrics_to_use: list,
    ) -> None:
        # Conversion from DEMETR language tag to SONAR language code
        self.language_codes = {
            "chinese_simple": "zho_Hans",  # Hans for Simplified script
            "czech": "ces_Latn",
            "french": "fra_Latn",
            "german": "deu_Latn",
            "hindi": "hin_Deva",
            "italian": "ita_Latn",
            "japanese": "jpn_Jpan",
            "polish": "pol_Latn",
            "russian": "rus_Cyrl",
            "spanish": "spa_Latn",
        }
        self.output_path = output_filepath
        self.demetr_root = demetr_root
        self.metrics_to_use = metrics_to_use

        colnames = ["category", *self.metrics_to_use]

        with open(self.output_path, "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(colnames)

        if "SacreBLEU" in self.metrics_to_use:
            self.sacre_bleu = SacreBLEUScore()
        if "BLASER_ref" in self.metrics_to_use:
            self.blaser_ref = BLASERRefScore()
        if "BLASER_qe" in self.metrics_to_use:
            self.blaser_qe = BLASERQEScore()
        if "COMET_ref" in self.metrics_to_use:
            self.comet_ref = COMETRefScore()
        if "COMET_qe" in self.metrics_to_use:
            self.comet_qe = COMETQEScore()
        if "ChrF" in self.metrics_to_use:
            self.chrf = ChrFScore(word_order=1)
        if "ChrF2" in self.metrics_to_use:
            self.chrf2 = ChrFScore(word_order=2)

        print(f"Using metrics {self.metrics_to_use}")

    def process_demetr_category(
        self,
        category: int,
        cat_fp: str,
        num_samples: int,
        reverse_accuracy: bool = False,
    ) -> None:
        curr_ds_path = os.path.join(self.demetr_root, cat_fp)

        # Load sentences into dataframe
        demetr_df = pd.read_json(curr_ds_path)

        ref_txts = demetr_df["eng_sent"]  # Human translation
        mt_txts = demetr_df["mt_sent"]  # Original machine translation
        src_txts = demetr_df["src_sent"]  # Foreign language source
        dfluent_txts = demetr_df["pert_sent"]  # Perturbed machine translation
        src_langs = demetr_df["lang_tag"]  # Source language
        blaser_lang_codes = src_langs.replace(self.language_codes)

        # Set up output arrays - typically (1000, n) where n is number of metrics
        # Two sets of results for each metric, one fluent and one disfluent
        mt_results = np.zeros((num_samples, len(self.metrics_to_use)))
        dis_results = np.zeros((num_samples, len(self.metrics_to_use)))

        for j, metric in enumerate(self.metrics_to_use):
            if metric == "COMET_ref":
                mt_results[:, j] = self.comet_ref.get_scores(
                    ref_txts, mt_txts, src_txts
                )
                dis_results[:, j] = self.comet_ref.get_scores(
                    ref_txts, dfluent_txts, src_txts
                )
            elif metric == "COMET_qe":
                mt_results[:, j] = self.comet_qe.get_scores(ref_txts, mt_txts, src_txts)
                dis_results[:, j] = self.comet_qe.get_scores(
                    ref_txts, dfluent_txts, src_txts
                )
            elif metric == "BLASER_ref":
                mt_results[:, j] = self.blaser_ref.get_scores(
                    ref_txts, mt_txts, src_txts, blaser_lang_codes
                )
                dis_results[:, j] = self.blaser_ref.get_scores(
                    ref_txts, dfluent_txts, src_txts, blaser_lang_codes
                )
            elif metric == "BLASER_qe":
                mt_results[:, j] = self.blaser_qe.get_scores(
                    mt_txts, src_txts, blaser_lang_codes
                )
                dis_results[:, j] = self.blaser_qe.get_scores(
                    dfluent_txts, src_txts, blaser_lang_codes
                )
            elif metric == "SacreBLEU":
                mt_results[:, j] = self.sacre_bleu.get_scores(ref_txts, mt_txts)
                dis_results[:, j] = self.sacre_bleu.get_scores(ref_txts, dfluent_txts)
            elif metric == "ChrF":
                mt_results[:, j] = self.chrf.get_scores(ref_txts, mt_txts)
                dis_results[:, j] = self.chrf.get_scores(ref_txts, dfluent_txts)
            elif metric == "ChrF2":
                mt_results[:, j] = self.chrf2.get_scores(ref_txts, mt_txts)
                dis_results[:, j] = self.chrf2.get_scores(ref_txts, dfluent_txts)
            else:
                print(f"Unknown metric {metric}")

        mask = mt_results > dis_results
        if reverse_accuracy:
            results = np.count_nonzero(~mask, axis=0)
        else:
            results = np.count_nonzero(mask, axis=0)

        results = results / num_samples * 100

        results_str = [category, *results]

        with open(self.output_path, "a") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(results_str)

    def process_demetr(
        self,
        samples_per_cat: int = 1000,
        cats_to_process: list | None = None,
    ) -> pd.DataFrame:
        if cats_to_process is None:
            cats_to_process = []

        # Get list of JSON files
        # Each file contains sentences for a single DEMETR category
        dataset_list = os.listdir(self.demetr_root)

        for ds in dataset_list:
            ds_cat = int(ds.split("_")[1].strip("id"))

            if ds_cat in cats_to_process or not cats_to_process:
                print(f"Processing input file {ds}")

                # Accuracy metric is reversed for category 35 as in this case the
                # reference text is passed as the disfluent translation and should
                # therefore score more highly
                reverse_acc = ds_cat == 35

                self.process_demetr_category(
                    ds_cat,
                    ds,
                    samples_per_cat,
                    reverse_acc,
                )
