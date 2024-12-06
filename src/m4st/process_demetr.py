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
            writer.writerow(
                [
                    "category",
                    "BLEU",
                    "SacreBLEU",
                    "BLASER_ref",
                    "BLASER_qe",
                    "COMET_ref",
                    "COMET_qe",
                ]
            )

    def get_accuracy_score(
        self,
        mt_scores: list,
        dfluent_scores: list,
        num_samples: int,
        reverse_accuracy: bool,
    ) -> float:

        mask = np.array(mt_scores) > np.array(dfluent_scores)
        result = np.count_nonzero(~mask) if reverse_accuracy else np.count_nonzero(mask)
        return result / num_samples * 100

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

        blaser_ref_mt = []
        blaser_ref_d = []

        blaser_qe_mt = []
        blaser_qe_d = []

        comet_ref_mt = []
        comet_ref_d = []

        comet_qe_mt = []
        comet_qe_d = []

        sacre_bleu = SacreBLEUScore()
        blaser = BLASERScore()
        comet = COMETScore()

        for sentence in json_data:

            ref_txt = sentence["eng_sent"]  # Human translation
            mt_txt = sentence["mt_sent"]  # Original machine translation
            src_text = sentence["src_sent"]  # Foreign language source
            dfluent_txt = sentence["pert_sent"]  # Perturbed machine translation
            src_lang = sentence["lang_tag"]  # Source language
            blaser_lang_code = self.language_codes[src_lang]

            # String-based metrics
            nltk_bleu_mt.append(nltk_bleu_score(ref_txt, mt_txt))
            nltk_bleu_d.append(nltk_bleu_score(ref_txt, dfluent_txt))
            sacre_bleu_mt.append(sacre_bleu.get_score(ref_txt, mt_txt))
            sacre_bleu_d.append(sacre_bleu.get_score(ref_txt, dfluent_txt))

            # Model-based metrics
            # BLASER-2.0
            blaser_ref_mt.append(
                blaser.blaser_ref_score(ref_txt, mt_txt, src_text, blaser_lang_code)
            )
            blaser_ref_d.append(
                blaser.blaser_ref_score(
                    ref_txt, dfluent_txt, src_text, blaser_lang_code
                )
            )
            blaser_qe_mt.append(
                blaser.blaser_qe_score(mt_txt, src_text, blaser_lang_code)
            )
            blaser_qe_d.append(
                blaser.blaser_qe_score(dfluent_txt, src_text, blaser_lang_code)
            )

            # COMET
            comet_ref_mt.append(comet.comet_ref_score(ref_txt, mt_txt, src_text))
            comet_ref_d.append(comet.comet_ref_score(ref_txt, dfluent_txt, src_text))
            comet_qe_mt.append(comet.comet_qe_score(mt_txt, src_text))
            comet_qe_d.append(comet.comet_qe_score(dfluent_txt, src_text))

        # Calculate accuracy as in DEMETR paper
        nltk_bleu_avg = self.get_accuracy_score(
            nltk_bleu_mt, nltk_bleu_d, num_samples, reverse_accuracy
        )
        sacre_bleu_avg = self.get_accuracy_score(
            sacre_bleu_mt, sacre_bleu_d, num_samples, reverse_accuracy
        )
        blaser_ref_avg = self.get_accuracy_score(
            blaser_ref_mt, blaser_ref_d, num_samples, reverse_accuracy
        )
        blaser_qe_avg = self.get_accuracy_score(
            blaser_qe_mt, blaser_qe_d, num_samples, reverse_accuracy
        )
        comet_ref_avg = self.get_accuracy_score(
            comet_ref_mt, comet_ref_d, num_samples, reverse_accuracy
        )
        comet_qe_avg = self.get_accuracy_score(
            comet_qe_mt, comet_qe_d, num_samples, reverse_accuracy
        )

        with open(self.output_path, "a") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(
                [
                    category,
                    nltk_bleu_avg,
                    sacre_bleu_avg,
                    blaser_ref_avg,
                    blaser_qe_avg,
                    comet_ref_avg,
                    comet_qe_avg,
                ]
            )

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
