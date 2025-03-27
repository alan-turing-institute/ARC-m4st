import json
import os

import evaluate
import numpy as np
from pandas import DataFrame

from m4st.metrics import Metric


class COMETScore(Metric):
    """Applies COMET metric using the evaluate library.
    All COMET models require the same input footprint, including QE versions. You can
    switch between different COMET models by providing the model argument.
    e.g. model="wmt21-comet-qe-mqmq", model="Unbabel/XCOMET-XXL".
    See https://huggingface.co/spaces/evaluate-metric/comet for more details.
    """

    def __init__(self, model: str = "wmt21-comet-mqm") -> None:
        self.model = model
        self.comet = evaluate.load("comet", self.model)

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"{self.model}_{input_fp}"
        sentence_ids = np.array(cat_data["id"])
        src_langs = list(cat_data["lang_tag"])

        mt_scores = self.comet.compute(
            predictions=cat_data["mt_sent"],
            references=cat_data["eng_sent"],
            sources=cat_data["src_sent"],
        )
        d_scores = self.comet.compute(
            predictions=cat_data["pert_sent"],
            references=cat_data["eng_sent"],
            sources=cat_data["src_sent"],
        )

        results = {}

        for i in range(len(mt_scores["scores"])):
            results[int(sentence_ids[[i]])] = {
                "source_language": src_langs[i],
                "mt_score": mt_scores["scores"][i],
                "disfluent_score": d_scores["scores"][i],
            }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
