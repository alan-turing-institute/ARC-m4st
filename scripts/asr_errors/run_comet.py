"""
Score a source and translation group with COMET Kiwi.
"""

from comet import download_model, load_from_checkpoint
from utils import (
    get_group,
    get_group_hypotheses,
    get_group_sources,
    get_reference,
    get_scores_path,
)


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
    group = get_group()
    sources = get_group_sources(group)
    hypotheses = get_group_hypotheses(group)
    reference = get_reference()
    scores_path = get_scores_path(group, "comet")

    model = COMETScore("Unbabel/wmt22-cometkiwi-da")

    ref_score = model([sources[-1]], [reference])
    print("REF SCORE:", ref_score)

    scores = model(sources, hypotheses)
    with open(scores_path, "w") as f:
        for score in scores:
            f.write(f"{score}\n")
