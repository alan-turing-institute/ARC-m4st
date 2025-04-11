import json
import os

from comet import download_model, load_from_checkpoint
from pandas import DataFrame, Series

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
        self.model_name = model.replace("/", "_")  # for save paths
        self.predict_kwargs = predict_kwargs  # passed to comet predict method

    def get_scores(
        self, references: Series, predictions: Series, sources: Series
    ) -> list:
        data = [
            {"src": s, "mt": mt, "ref": r}
            for s, mt, r in zip(references, predictions, sources, strict=True)
        ]
        return self.comet.predict(data, **self.predict_kwargs)["scores"]

    def process_demetr_cat(
        self, cat_data: DataFrame, output_path: str | os.PathLike, input_fp: str
    ) -> None:
        ref_txts = cat_data["eng_sent"]  # Human translation
        mt_txts = cat_data["mt_sent"]  # Original machine translation
        dfluent_txts = cat_data["pert_sent"]  # Perturbed machine translation
        src_txts = cat_data["src_sent"]  # Source (original) text
        src_langs = cat_data["lang_tag"]  # Source language
        sentence_ids = cat_data["id"]

        mt_scores = self.get_scores(ref_txts, mt_txts, src_txts)
        d_scores = self.get_scores(ref_txts, dfluent_txts, src_txts)

        results = {
            index: {
                "source_language": lang,
                "mt_score": mt_score,
                "disfluent_score": d_score,
            }
            for index, lang, mt_score, d_score in zip(
                sentence_ids, src_langs, mt_scores, d_scores, strict=True
            )
        }

        output_file = f"{self.model_name}_{input_fp}"
        with open(os.path.join(output_path, output_file), "w+") as file_to_write:
            json.dump(results, file_to_write)
