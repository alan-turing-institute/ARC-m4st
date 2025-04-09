import json
import os

import numpy as np
from comet import download_model, load_from_checkpoint
from pandas import DataFrame

from m4st.metrics import Metric


class COMETScore(Metric):
    """Applies COMET metric using the evaluate library.
    All COMET models require the same input footprint, including QE versions. You can
    switch between different COMET models by providing the model argument.
    e.g. model="wmt21-comet-qe-mqmq", model="Unbabel/XCOMET-XXL".
    See https://huggingface.co/spaces/evaluate-metric/comet for more details.
    """

    def __init__(self, model: str = "Unbabel/wmt22-comet-da", **predict_kwargs) -> None:
        self.comet = load_from_checkpoint(download_model(model))
        self.model = model.replace("/", "_")  # for save paths
        self.predict_kwargs = predict_kwargs  # passed to comet predict method

    def get_scores(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        output_file = f"{self.model}_{input_fp}"
        sentence_ids = np.array(cat_data["id"])
        src_langs = list(cat_data["lang_tag"])

        fluent_data = [
            {"src": row["src_sent"], "mt": row["mt_sent"], "ref": row["eng_sent"]}
            for _, row in cat_data.iterrows()
        ]
        disfluent_data = [
            {"src": row["src_sent"], "mt": row["pert_sent"], "ref": row["eng_sent"]}
            for _, row in cat_data.iterrows()
        ]

        fluent_scores = self.comet.predict(fluent_data, **self.predict_kwargs)["scores"]
        disfluent_scores = self.comet.predict(disfluent_data, **self.predict_kwargs)[
            "scores"
        ]

        results = {}
        for i in range(len(fluent_scores)):
            results[int(sentence_ids[[i]])] = {
                "source_language": src_langs[i],
                "mt_score": fluent_scores[i],
                "disfluent_score": disfluent_scores[i],
            }

        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
