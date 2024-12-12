"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import csv
import os

import numpy as np
import pandas as pd

from m4st.metrics import BLASERScore, COMETScore, SacreBLEUScore, nltk_bleu_score
from m4st.utils import load_json


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

        colnames = ["category", *metrics_to_use]

        with open(self.output_path, "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(colnames)

        if "Sacre_BLEU" in metrics_to_use:
            self.sacre_bleu = SacreBLEUScore()
        if "BLASER_ref" in metrics_to_use or "BLASER_qe" in metrics_to_use:
            self.blaser = BLASERScore()
        if "COMET_ref" in metrics_to_use or "COMET_qe" in metrics_to_use:
            self.comet = COMETScore()

    def process_demetr_category(
        self,
        category: int,
        cat_fp: str,
        num_samples: int,
        reverse_accuracy: bool = False,
    ) -> list:

        curr_ds_path = os.path.join(self.demetr_root, cat_fp)
        json_data = load_json(curr_ds_path)

        mt_results = np.zeros((num_samples, len(self.metrics_to_use)))
        dis_results = np.zeros((num_samples, len(self.metrics_to_use)))

        for i, sentence in enumerate(json_data):

            ref_txt = sentence["eng_sent"]  # Human translation
            mt_txt = sentence["mt_sent"]  # Original machine translation
            src_text = sentence["src_sent"]  # Foreign language source
            dfluent_txt = sentence["pert_sent"]  # Perturbed machine translation
            src_lang = sentence["lang_tag"]  # Source language
            blaser_lang_code = self.language_codes[src_lang]

            for j, metric in enumerate(self.metrics_to_use):
                if metric == "BLEU":
                    mt_results[i, j] = nltk_bleu_score(ref_txt, mt_txt)
                    dis_results[i, j] = nltk_bleu_score(ref_txt, dfluent_txt)
                elif metric == "Sacre_BLEU":
                    mt_results[i, j] = self.sacre_bleu.get_score(ref_txt, mt_txt)
                    dis_results[i, j] = self.sacre_bleu.get_score(ref_txt, dfluent_txt)
                elif metric == "BLASER_ref":
                    mt_results[i, j] = self.blaser.blaser_ref_score(
                        ref_txt, mt_txt, src_text, blaser_lang_code
                    )
                    dis_results[i, j] = self.blaser.blaser_ref_score(
                        ref_txt, dfluent_txt, src_text, blaser_lang_code
                    )
                elif metric == "BLASER_qe":
                    mt_results[i, j] = self.blaser.blaser_qe_score(
                        mt_txt, src_text, blaser_lang_code
                    )
                    dis_results[i, j] = self.blaser.blaser_qe_score(
                        dfluent_txt, src_text, blaser_lang_code
                    )
                elif metric == "COMET_ref":
                    mt_results[i, j] = self.comet.comet_ref_score(
                        ref_txt, mt_txt, src_text
                    )
                    dis_results[i, j] = self.comet.comet_ref_score(
                        ref_txt, dfluent_txt, src_text
                    )
                elif metric == "COMET_qe":
                    mt_results[i, j] = self.comet.comet_qe_score(mt_txt, src_text)
                    dis_results[i, j] = self.comet.comet_qe_score(dfluent_txt, src_text)
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
                reverse_acc = ds_cat == 35

                self.process_demetr_category(
                    ds_cat,
                    ds,
                    samples_per_cat,
                    reverse_acc,
                )
