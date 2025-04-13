from comet import download_model, load_from_checkpoint

from m4st.metrics import Metric, TranslationDataset


class COMETScore(Metric):
    """Applies COMET metric using the evaluate library.
    All COMET models require the same input footprint, including QE versions. You can
    switch between different COMET models by providing the model argument.
    e.g. model="wmt21-comet-qe-mqmq", model="Unbabel/XCOMET-XXL".
    See https://huggingface.co/spaces/evaluate-metric/comet for more details.
    """

    def __init__(self, model: str = "Unbabel/wmt22-comet-da", **predict_kwargs) -> None:
        self.comet = load_from_checkpoint(download_model(model))
        self.name = model.replace("/", "_")  # for save paths
        self.predict_kwargs = predict_kwargs  # passed to comet predict method
        if self.comet.hparams.class_identifier == "regression_metric":
            # older COMET models using the RegressionMetric class don't spescify
            # input_segments in hparams
            self.comet_req_inputs = ["src", "mt", "ref"]
        else:
            self.comet_req_inputs = self.comet.hparams.input_segments
        field_mapping = {"src": "source", "mt": "prediction", "ref": "reference"}
        self.data_req_inputs = [field_mapping[f] for f in self.comet_req_inputs]
        # only preserve mapping for fields required by the model
        self.field_mapping = dict(
            zip(self.comet_req_inputs, self.data_req_inputs, strict=True)
        )

    def get_scores(self, dataset: TranslationDataset) -> list[float]:
        self.check_dataset_compatible(dataset)

        comet_inputs = dataset.to_samples(self.field_mapping)
        return self.comet.predict(comet_inputs, **self.predict_kwargs)["scores"]
