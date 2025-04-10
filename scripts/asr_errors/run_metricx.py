"""
Score a source and translation group with MetricX QE.
"""

import json
import tempfile

import datasets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from utils import (
    get_group,
    get_group_hypotheses,
    get_group_sources,
    get_reference,
    get_scores_path,
)

from m4st.metrics.metricx import MT5ForRegression


def get_dataset(
    input_file: str,
    tokenizer,
    max_input_length: int,
    is_qe: bool,
    device: str | torch.device | None = None,
):
    """Gets the test dataset for prediction.

    If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
    If it is false, there must be "hypothesis" and "reference" fields.

    Args:
      input_file: The path to the jsonl input file.
      tokenizer: The tokenizer to use.
      max_input_length: The maximum input sequence length.
      device: The ID of the device to put the PyTorch tensors on.
      is_qe: Indicates whether the metric is a QE metric or not.

    Returns:
      The dataset.
    """

    def _make_input(example):
        if is_qe:
            example["input"] = (
                "source: " + example["source"] + " candidate: " + example["hypothesis"]
            )
        else:
            example["input"] = (
                "source: "
                + example["source"]
                + " candidate: "
                + example["hypothesis"]
                + " reference: "
                + example["reference"]
            )
        return example

    def _tokenize(example):
        return tokenizer(
            example["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

    def _remove_eos(example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    ds = datasets.load_dataset("json", data_files={"test": input_file})
    ds = ds.map(_make_input)
    ds = ds.map(_tokenize)
    ds = ds.map(_remove_eos)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=device,
        output_all_columns=True,
    )
    return ds


metricx_tokenizers = {
    "google/metricx-24-hybrid-xxl-v2p6": "google/mt5-xxl",
    "google/metricx-24-hybrid-xl-v2p6": "google/mt5-xl",
    "google/metricx-24-hybrid-large-v2p6": "google/mt5-large",
    "google/metricx-24-hybrid-xxl-v2p6-bfloat16": "google/mt5-xxl",
    "google/metricx-24-hybrid-xl-v2p6-bfloat16": "google/mt5-xl",
    "google/metricx-24-hybrid-large-v2p6-bfloat16": "google/mt5-large",
}


class MetricXScore:
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
        self, sources: list[str], hypotheses: list[str], references: list[str] | None
    ) -> Dataset:
        if references is None:
            references = [""] * len(sources)
        with tempfile.TemporaryDirectory() as tempdir:
            fname = f"{tempdir}/data.json"
            with open(fname, "a") as f:
                for src, hyp, ref in zip(sources, hypotheses, references, strict=False):
                    f.write(
                        json.dumps(
                            {
                                "source": src,
                                "hypothesis": hyp,
                                "reference": ref,
                            }
                        )
                    )
            return get_dataset(fname, self.tokenizer, self.max_input_length, self.qe)

    def compute(self, ds: Dataset) -> list[float]:
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                per_device_eval_batch_size=self.batch_size,
                use_cpu=True,
            )
            trainer = Trainer(model=self.model, args=training_args)
            predictions, _, _ = trainer.predict(test_dataset=ds["test"])

        return [float(pred) for pred in predictions]

    def __call__(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str] | None = None,
    ) -> list[float]:
        ds = self.preprocess(sources, hypotheses, references)
        return self.compute(ds)


if __name__ == "__main__":
    group = get_group()
    sources = get_group_sources(group)
    hypotheses = get_group_hypotheses(group)
    reference = get_reference()
    scores_path = get_scores_path(group, "comet")

    model = MetricXScore("google/metricx-24-hybrid-large-v2p6", qe=True)

    ref_score = model([sources[-1]], [reference])
    print("REF SCORE:", ref_score)

    scores = model(sources, hypotheses)
    with open(scores_path, "w") as f:
        for score in scores:
            f.write(f"{score}\n")
