from comet import download_model, load_from_checkpoint


class COMETScore:
    """Applies COMET metric using the evaluate library.
    All COMET models require the same input footprint, including QE versions. You can
    switch between different COMET models by providing the model argument.
    e.g. model="wmt21-comet-qe-mqmq", model="Unbabel/XCOMET-XXL".
    See https://huggingface.co/spaces/evaluate-metric/comet for more details.
    """

    def __init__(self, model: str = "wmt21-comet-mqm") -> None:
        # Choose your model from Hugging Face Hub
        model_path = download_model(model)
        # or for example:
        # model_path = download_model("Unbabel/wmt22-comet-da")

        # Load the model checkpoint:
        self.comet = load_from_checkpoint(model_path)

    def __call__(self, sources: list[str], hypotheses: list[str]) -> list[float]:
        data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses, strict=False)]
        return self.comet.predict(data, batch_size=8, gpus=0, num_workers=8).scores


if __name__ == "__main__":
    model = COMETScore("Unbabel/wmt22-cometkiwi-da")

    sources = []
    hypotheses = []
    for i in range(331):
        with open(f"data/source_merged/merged_source_{i}.txt") as f:
            sources.append(f.read())
        with open(f"data/translation_merged/merged_translation_{i}.txt") as f:
            hypotheses.append(f.read())

    with open("data/reference.txt") as f:
        reference = f.read()

    ref_score = model([sources[-1]], [reference])
    print("REF SCORE:", ref_score)

    scores = model(sources, hypotheses)
    with open("data/comet_merged_scores.txt", "w") as f:
        for score in scores:
            f.write(f"{score}\n")
