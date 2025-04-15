"""
Script for running BLEU, SacreBLEU, and BLASER 2.0 on the DEMETR dataset.
"""

import json
import os

import pandas as pd

from m4st.metrics import Metric, TranslationDataset
from m4st.metrics.blaser import BLASERScore
from m4st.metrics.comet import COMETScore
from m4st.metrics.metricx import MetricXScore
from m4st.metrics.string import BLEUScore, ChrFScore


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
        comet_model_str: str,
        metricx_model_str: str,
    ) -> None:
        self.output_dir = output_dir
        self.demetr_root = demetr_root
        self.metrics_to_use = metrics_to_use
        self.comet_model_str = comet_model_str
        self.metricx_model_str = metricx_model_str
        self.metrics_to_use = metrics_to_use

        # mapping DEMETR language codes to SONAR/BLASER language codes.
        # e.g. DEMETR may specify source language as "french" which requires the code
        # "fra_Latn" for SONAR embedding generation.
        self.demetr_to_sonar_lang_mapping = {
            "chinese_simple": "zho_Hans",
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
        self.sonar_to_demetr_lang_mapping = {
            v: k for k, v in self.demetr_to_sonar_lang_mapping.items()
        }
        self.target_language = "eng_Latn"

        print(f"Using metrics {self.metrics_to_use}")

    def setup_metric(self, metric_specifier: str) -> Metric:
        if metric_specifier == "BLEU":
            return BLEUScore()
        if metric_specifier == "ChrF":
            return ChrFScore(word_order=1)
        if metric_specifier == "ChrF2":
            return ChrFScore(word_order=2)
        if metric_specifier == "COMET":
            return COMETScore(model=self.comet_model_str)
        if metric_specifier == "MetricX_ref":
            return MetricXScore(qe=False, model=self.metricx_model_str)
        if metric_specifier == "MetricX_qe":
            return MetricXScore(qe=True, model=self.metricx_model_str)
        if metric_specifier == "BLASER_ref":
            return BLASERScore(qe=False)
        if metric_specifier == "BLASER_qe":
            return BLASERScore(qe=True)

        msg = f"Unknown metric specifier {metric_specifier}"
        raise ValueError(msg)

    def save_metric_cat_scores(
        self,
        metric: Metric,
        mt_ds: TranslationDataset,
        disfluent_ds: TranslationDataset,
        cat_fp: str,
    ) -> None:
        """
        Compute scores for a single metric on a single category of the DEMETR dataset.
        """

        mt_scores = metric.get_scores(mt_ds)
        disfluent_scores = metric.get_scores(disfluent_ds)

        # results with SONAR language codes converted back to DEMETR codes
        results = {
            index: {
                "source_language": self.sonar_to_demetr_lang_mapping[lang],
                "mt_score": mt_score,
                "disfluent_score": d_score,
            }
            for index, lang, mt_score, d_score in zip(
                mt_ds.index,
                mt_ds.source_language,
                mt_scores,
                disfluent_scores,
                strict=True,
            )
        }

        output_file = os.path.join(self.output_dir, f"{metric.name}_{cat_fp}")
        with open(output_file, "w+") as file_to_write:
            json.dump(results, file_to_write)

    def process_demetr_category(
        self,
        cat_fp: str,
    ) -> None:
        curr_ds_path = os.path.join(self.demetr_root, cat_fp)

        # Load sentences into dataframe
        demetr_df = pd.read_json(curr_ds_path)

        # target always english
        target_language = [self.target_language] * len(demetr_df)
        # Convert DEMETR source language codes to SONAR/BLASER language codes.
        # (BLASER is the only metric the requires language codes)
        source_language = [
            self.demetr_to_sonar_lang_mapping[lang] for lang in demetr_df["lang_tag"]
        ]

        mt_ds = TranslationDataset(
            reference=demetr_df["eng_sent"],  # Human translation
            prediction=demetr_df["mt_sent"],  # Original machine translation
            source=demetr_df["src_sent"],  # Source (original) text
            source_language=source_language,  # Source language
            target_language=target_language,  # Target language
            index=demetr_df["id"],  # Sentence ID
        )
        disfluent_ds = TranslationDataset(
            reference=demetr_df["eng_sent"],  # Human translation
            prediction=demetr_df["pert_sent"],  # Perturbed machine translation
            source=demetr_df["src_sent"],  # Source (original) text
            source_language=source_language,  # Source language
            target_language=target_language,  # Target language
            index=demetr_df["id"],  # Sentence ID
        )

        for metric_specifier in self.metrics_to_use:  # type: ignore[has-type]
            print(metric_specifier)
            metric = self.setup_metric(metric_specifier)
            self.save_metric_cat_scores(metric, mt_ds, disfluent_ds, cat_fp)

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
