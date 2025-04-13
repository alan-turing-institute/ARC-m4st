import evaluate
from tqdm import tqdm

from m4st.metrics import Metric, TranslationDataset


class ChrFScore(Metric):
    """Applies ChrF/++ from the evaluate library.
    When word_order=0 (default) computes original ChrF metric without including word
    n-grams. When word_order=2, computes ChrF++. The DEMETR paper refers to ChrF++
    as ChrF2.For more details see https://huggingface.co/spaces/evaluate-metric/chrf"""

    def __init__(self, word_order: int = 0) -> None:
        self.chrf = evaluate.load("chrf")
        self.word_order = word_order
        self.name = f"ChrF{self.word_order}" if word_order > 0 else "ChrF"
        self.data_req_inputs = ["prediction", "reference"]

    def get_scores(self, dataset: TranslationDataset) -> list[float]:
        self.check_dataset_compatible(dataset)

        return [
            self.chrf.compute(
                predictions=[sample["prediction"]],
                references=[[sample["reference"]]],
                word_order=self.word_order,
                eps_smoothing=True,
            )["score"]
            for sample in tqdm(dataset)
        ]


class BLEUScore(Metric):
    """Applies SacreBleu from the evaluate library."""

    def __init__(self) -> None:
        self.name = "BLEU"
        self.bleu = evaluate.load("sacrebleu")
        self.data_req_inputs = ["prediction", "reference"]

    def get_scores(self, dataset: TranslationDataset) -> list[float]:
        self.check_dataset_compatible(dataset)

        return [
            self.bleu.compute(
                predictions=[sample["prediction"]], references=[[sample["reference"]]]
            )["score"]
            for sample in tqdm(dataset)
        ]
