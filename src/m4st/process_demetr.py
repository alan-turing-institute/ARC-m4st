"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import csv
import os

import numpy as np
import pandas as pd

from m4st.metrics import BLASERScore, SacreBLEUScore, nltk_bleu_score
from m4st.utils import load_json


class ProcessDEMETR:
    def __init__(
        self, output_filepath: os.PathLike | str, demetr_root: os.PathLike | str
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

        with open(self.output_path, "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["category", "BLEU", "SacreBLEU", "BLASER"])

    def process_demetr_category(
        self,
        category: int,
        cat_fp: str,
        num_samples: int,
        reverse_accuracy: bool = False,
    ) -> list:

        curr_ds_path = os.path.join(self.demetr_root, cat_fp)
        json_data = load_json(curr_ds_path)

        nltk_bleu_mt = []
        nltk_bleu_d = []

        sacre_bleu_mt = []
        sacre_bleu_d = []

        blaser_mt = []
        blaser_d = []

        sacre_bleu = SacreBLEUScore()
        blaser = BLASERScore()

        for sentence in json_data:

            ref_txt = sentence["eng_sent"]  # Human translation
            mt_txt = sentence["mt_sent"]  # Original machine translation
            src_text = sentence["src_sent"]  # Foreign language source
            dfluent_txt = sentence["pert_sent"]  # Perturbed machine translation
            src_lang = sentence["lang_tag"]  # Source language
            blaser_lang_code = self.language_codes[src_lang]

            nltk_bleu_mt.append(nltk_bleu_score(ref_txt, mt_txt))
            nltk_bleu_d.append(nltk_bleu_score(ref_txt, dfluent_txt))
            sacre_bleu_mt.append(sacre_bleu.get_score(ref_txt, mt_txt))
            sacre_bleu_d.append(sacre_bleu.get_score(ref_txt, dfluent_txt))
            blaser_mt.append(
                blaser.blaser_ref_score(ref_txt, mt_txt, src_text, blaser_lang_code)
            )
            blaser_d.append(
                blaser.blaser_ref_score(
                    ref_txt, dfluent_txt, src_text, blaser_lang_code
                )
            )

        nltk_bleu_mask = np.array(nltk_bleu_mt) > np.array(nltk_bleu_d)
        sacre_bleu_mask = np.array(sacre_bleu_mt) > np.array(sacre_bleu_d)
        blaser_mask = np.array(blaser_mt) > np.array(blaser_d)

        if reverse_accuracy:
            nltk_bleu_res = np.count_nonzero(~nltk_bleu_mask)
            sacre_bleu_res = np.count_nonzero(~sacre_bleu_mask)
            blaser_res = np.count_nonzero(~blaser_mask)
        else:
            nltk_bleu_res = np.count_nonzero(nltk_bleu_mask)
            sacre_bleu_res = np.count_nonzero(sacre_bleu_mask)
            blaser_res = np.count_nonzero(blaser_mask)

        nltk_bleu_avg = nltk_bleu_res / num_samples * 100
        sacre_bleu_avg = sacre_bleu_res / num_samples * 100
        blaser_avg = blaser_res / num_samples * 100

        with open(self.output_path, "a") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow([category, nltk_bleu_avg, sacre_bleu_avg, blaser_avg])

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
            print(f"Processing input file {ds}")
            ds_cat = int(ds.split("_")[1].strip("id"))

            if ds_cat in cats_to_process or not cats_to_process:

                reverse_acc = ds_cat == 35

                self.process_demetr_category(ds_cat, ds, samples_per_cat, reverse_acc)
