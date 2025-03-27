import json
import os
import tempfile

import numpy as np
from datasets import Dataset
from metricx import MT5ForRegression, get_dataset
from pandas import DataFrame
from transformers import AutoTokenizer, Trainer, TrainingArguments

from m4st.metrics import Metric

metricx_tokenizers = {
    "google/metricx-24-hybrid-xxl-v2p6": "google/mt5-xxl",
    "google/metricx-24-hybrid-xl-v2p6": "google/mt5-xl",
    "google/metricx-24-hybrid-large-v2p6": "google/mt5-large",
    "google/metricx-24-hybrid-xxl-v2p6-bfloat16": "google/mt5-xxl",
    "google/metricx-24-hybrid-xl-v2p6-bfloat16": "google/mt5-xl",
    "google/metricx-24-hybrid-large-v2p6-bfloat16": "google/mt5-large",
}


class MetricXScore(Metric):
    """Applies MetricX 2024: https://github.com/google-research/metricx"""

    def __init__(
        self,
        model: str = "google/metricx-24-hybrid-xl-v2p6",
        max_input_length: int = 1536,
        batch_size: int = 1,
        qe: bool = False,
    ) -> None:
        if model not in metricx_tokenizers:
            msg = f"{model} is not a known MetricX model."
            raise KeyError(msg)

        self.tokenizer = AutoTokenizer.from_pretrained(metricx_tokenizers[model])
        self.model = MT5ForRegression.from_pretrained(model)
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.qe = qe

        self.model_name = model.replace("/", "_")  # for output file paths
        if self.qe:
            self.model_name += "_qe"
        else:
            self.model_name += "_ref"

    def preprocess(
        self, cat_data: DataFrame, src_col: str, pred_col: str, ref_col: str
    ) -> Dataset:
        with tempfile.TemporaryDirectory() as tempdir:
            fname = f"{tempdir}/cat_data.json"
            with open(fname, "a") as f:
                for _, row in cat_data.iterrows():
                    f.write(
                        json.dumps(
                            {
                                "source": row[src_col],
                                "hypothesis": row[pred_col],
                                "reference": row[ref_col],
                            }
                        )
                    )
            return get_dataset(fname, self.tokenizer, self.max_input_length, self.qe)

    def compute(self, ds: Dataset) -> list[float]:
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                per_device_eval_batch_size=self.batch_size,
            )
            trainer = Trainer(model=self.model, args=training_args)
            predictions, _, _ = trainer.predict(test_dataset=ds["test"])

        return [float(pred) for pred in predictions]

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"{self.model_name}_{input_fp}"
        sentence_ids = np.array(cat_data["id"])
        src_langs = list(cat_data["lang_tag"])

        mt_data = self.preprocess(
            cat_data, src_col="src_sent", pred_col="mt_sent", ref_col="eng_sent"
        )
        mt_scores = self.compute(mt_data)

        d_data = self.preprocess(
            cat_data, src_col="src_sent", pred_col="pert_sent", ref_col="eng_sent"
        )
        d_scores = self.compute(d_data)

        results = {}

        for i in range(len(mt_scores)):
            results[int(sentence_ids[[i]])] = {
                "source_language": src_langs[i],
                "mt_score": mt_scores[i],
                "disfluent_score": d_scores[i],
            }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
