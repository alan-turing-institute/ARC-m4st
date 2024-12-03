"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import os

import numpy as np
import pandas as pd

from m4st.metrics import BLASERScore, SacreBLEUScore, nltk_bleu_score
from m4st.utils import load_json


def process_demetr_category(
    cat_fp: str,
    dataset_root: os.PathLike | str,
    reverse_accuracy: bool = False,
) -> list:

    curr_ds_path = os.path.join(dataset_root, cat_fp)
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

        nltk_bleu_mt.append(nltk_bleu_score(ref_txt, mt_txt))
        nltk_bleu_d.append(nltk_bleu_score(ref_txt, dfluent_txt))
        sacre_bleu_mt.append(sacre_bleu.get_score(ref_txt, mt_txt))
        sacre_bleu_d.append(sacre_bleu.get_score(ref_txt, dfluent_txt))
        blaser_mt.append(blaser.blaser_score(ref_txt, mt_txt, src_text))
        blaser_d.append(blaser.blaser_score(ref_txt, dfluent_txt, src_text))

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

    return nltk_bleu_res, sacre_bleu_res, blaser_res


def process_demetr(
    dataset_dir: os.PathLike | str,
    samples_per_cat: int = 1000,
    cats_to_process: list | None = None,
) -> pd.DataFrame:

    if cats_to_process is None:
        cats_to_process = []
    sacre_bleu_res = []
    nltk_bleu_res = []
    blaser_res = []
    cat = []

    # Get list of JSON files
    # Each file contains sentences for a single DEMETR category
    dataset_list = os.listdir(dataset_dir)

    for ds in dataset_list:
        ds_cat = int(ds.split("_")[1].strip("id"))

        if ds_cat in cats_to_process or not cats_to_process:

            cat.append(ds_cat)

            reverse_acc = ds_cat == 35

            metrics = process_demetr_category(ds, dataset_dir, reverse_acc)
            nltk_bleu_res.append(metrics[0] / samples_per_cat * 100)
            sacre_bleu_res.append(metrics[1] / samples_per_cat * 100)
            blaser_res.append(metrics[2] / samples_per_cat * 100)

    return pd.DataFrame(
        {
            "category": cat,
            "BLEU": nltk_bleu_res,
            "SacreBLEU": sacre_bleu_res,
            "BLASER": blaser_res,
        }
    )
