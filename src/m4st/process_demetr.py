"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import os
import typing

import pandas as pd


class ProcessDEMETR:
    """Run the specified metrics over the DEMETR dataset from
    https://github.com/marzenakrp/demetr.

    output_dir --               the directory for storing output JSON files. One JSON
                                file will be produced for each DEMETR category, for each
                                metric.
    demetr_root --              root directory for the DEMETR dataset. This can be
                                downloaded from https://github.com/marzenakrp/demetr.
                                The argument should point to the directory containing
                                the input JSON files.
    metrics_to_use --           list of metrics to run. Must be one or more of
                                COMET, BLASER_ref, BLASER_qe, BLEU, ChrF, ChrF2.
    blaser_lang_code_config -- config YAML mapping DEMETR language codes to SONAR/BLASER
                                language codes. e.g. DEMETR may specify source language
                                as "french" which requires the code "fra_Latn" for SONAR
                                embedding generation.
    comet_model_str --          COMET model to use, e.g. Unlabel/wmt22-comet-da
                                (default), Unbabel/XCOMET-XL.

    metricx_model_str --        MetricX 24 model to use, e.g.
                                google/metricx-24-hybrid-xl-v2p6" (default),
    """

    def __init__(
        self,
        output_dir: os.PathLike | str,
        demetr_root: os.PathLike | str,
        metrics_to_use: list,
        blaser_lang_code_config: os.PathLike | str,
        comet_model_str: str,
        metricx_model_str: str,
    ) -> None:
        self.output_dir = output_dir
        self.demetr_root = demetr_root
        self.metrics_to_use = metrics_to_use
        self.blaser_lang_code_config = blaser_lang_code_config
        self.comet_model_str = comet_model_str
        self.metricx_model_str = metricx_model_str

        self.setup_metrics()

        print(f"Using metrics {self.metrics_to_use}")

    @typing.no_type_check
    def setup_metrics(self) -> None:
        metrics = []

        if "BLEU" in self.metrics_to_use:
            from m4st.metrics.string import BLEUScore

            metrics.append(BLEUScore())

        if "BLASER_ref" in self.metrics_to_use:
            from m4st.metrics.blaser import BLASERRefScore

            metrics.append(
                BLASERRefScore(lang_code_config=self.blaser_lang_code_config)
            )

        if "ChrF" in self.metrics_to_use:
            from m4st.metrics.string import ChrFScore

            metrics.append(ChrFScore(word_order=1))

        if "ChrF2" in self.metrics_to_use:
            from m4st.metrics.string import ChrFScore

            metrics.append(ChrFScore(word_order=2))

        if "BLASER_qe" in self.metrics_to_use:
            from m4st.metrics.blaser import BLASERQEScore

            metrics.append(BLASERQEScore(lang_code_config=self.blaser_lang_code_config))

        if "COMET" in self.metrics_to_use:
            from m4st.metrics.comet import COMETScore

            metrics.append(COMETScore(model=self.comet_model_str))

        if "MetricX_ref" in self.metrics_to_use:
            from m4st.metrics.metricx import MetricXScore

            metrics.append(MetricXScore(qe=False, model=self.metricx_model_str))

        if "MetricX_qe" in self.metrics_to_use:
            from m4st.metrics.metricx import MetricXScore

            metrics.append(MetricXScore(qe=True, model=self.metricx_model_str))

        self.metrics = metrics

    def process_demetr_category(
        self,
        cat_fp: str,
    ) -> None:
        curr_ds_path = os.path.join(self.demetr_root, cat_fp)

        # Load sentences into dataframe
        demetr_df = pd.read_json(curr_ds_path)

        for metric in self.metrics:  # type: ignore[has-type]
            metric.process_demetr_cat(demetr_df, self.output_dir, cat_fp)

    def process_demetr(
        self,
        cats_to_process: list | None = None,
    ) -> pd.DataFrame:
        """Iterates over the input files, processing each category in turn.

        cats_to_process -- list of DEMETR categories to process. These are numbered
                            1 to 35, and can be found in the DEMETR paper
                            (https://arxiv.org/abs/2210.13746). Defaults to running
                            over all categories.
        """
        if cats_to_process is None:
            cats_to_process = []

        # Get list of JSON files
        # Each file contains sentences for a single DEMETR category
        dataset_list = os.listdir(self.demetr_root)
        print(f"Found {len(dataset_list)} input files")

        for ds in dataset_list:
            ds_cat = int(ds.split("_")[1].strip("id"))

            if ds_cat in cats_to_process or not cats_to_process:
                print(f"Processing input file {ds}")

                self.process_demetr_category(ds)
