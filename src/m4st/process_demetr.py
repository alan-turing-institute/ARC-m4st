"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import os
import typing

import pandas as pd

from m4st.metrics import (
    BLASERQEScore,
    BLASERRefScore,
    BleuScore,
    ChrFScore,
    COMETQEScore,
    COMETRefScore,
)


class ProcessDEMETR:
    def __init__(
        self,
        output_dir: os.PathLike | str,
        demetr_root: os.PathLike | str,
        metrics_to_use: list,
    ) -> None:
        self.output_dir = output_dir
        self.demetr_root = demetr_root
        self.metrics_to_use = metrics_to_use

        self.setup_metrics()

        print(f"Using metrics {self.metrics_to_use}")

    @typing.no_type_check
    def setup_metrics(self) -> None:
        metrics = []

        if "Bleu" in self.metrics_to_use:
            metrics.append(BleuScore())
        if "BLASER_ref" in self.metrics_to_use:
            metrics.append(BLASERRefScore())
        if "BLASER_qe" in self.metrics_to_use:
            metrics.append(BLASERQEScore())
        if "COMET_ref" in self.metrics_to_use:
            metrics.append(COMETRefScore())
        if "COMET_qe" in self.metrics_to_use:
            metrics.append(COMETQEScore())
        if "ChrF" in self.metrics_to_use:
            metrics.append(ChrFScore(word_order=1))
        if "ChrF2" in self.metrics_to_use:
            metrics.append(ChrFScore(word_order=2))

        self.metrics = metrics

    def process_demetr_category(
        self,
        cat_fp: str,
    ) -> None:
        curr_ds_path = os.path.join(self.demetr_root, cat_fp)

        # Load sentences into dataframe
        demetr_df = pd.read_json(curr_ds_path)

        for metric in self.metrics:  # type: ignore[has-type]
            metric.get_scores(demetr_df, self.output_dir, cat_fp)

    def process_demetr(
        self,
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

                self.process_demetr_category(ds)
