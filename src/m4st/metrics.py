import evaluate
from nltk.translate.bleu_score import sentence_bleu
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model


def nltk_bleu_score(reference: str, prediction: str) -> float:
    return sentence_bleu([reference.split()], prediction.split())


class SacreBLEUScore:
    """Applies SacreBLEU from the evaluate library."""

    def __init__(self) -> None:
        self.bleu = evaluate.load("sacrebleu")

    def get_score(self, reference: str, prediction: str) -> float:

        score = self.bleu.compute(predictions=[prediction], references=[[reference]])
        return score["score"]


class BLASERScore:
    """Initialises and applies the BLASER 2.0 metric from the SONAR library."""

    def __init__(self) -> None:
        self.blaser_ref = load_blaser_model("blaser_2_0_ref").eval()
        self.blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )

    def blaser_ref_score(
        self, reference: str, prediction: str, source: str, source_lang_code: str
    ) -> float:

        src_embs = self.text_embedder.predict([source], source_lang=source_lang_code)
        ref_embs = self.text_embedder.predict([reference], source_lang="eng_Latn")
        mt_embs = self.text_embedder.predict([prediction], source_lang="eng_Latn")

        return self.blaser_ref(src=src_embs, ref=ref_embs, mt=mt_embs).item()

    def blaser_qe_score(
        self, prediction: str, source: str, source_lang_code: str
    ) -> float:

        src_embs = self.text_embedder.predict([source], source_lang=source_lang_code)
        mt_embs = self.text_embedder.predict([prediction], source_lang="eng_Latn")

        return self.blaser_qe(src=src_embs, mt=mt_embs).item()


class COMETScore:
    """Applies COMET and COMET-QE metrics from the evaluate library."""

    def __init__(self) -> None:

        self.comet = evaluate.load("comet", model="wmt21-comet-mqm")
        self.comet_qe = evaluate.load("comet", model="wmt21-comet-qe-mqm")

    def comet_ref_score(self, reference: str, prediction: str, source: str) -> float:

        score = self.comet.compute(
            predictions=[prediction],
            references=[reference],
            sources=[source],
        )
        # Note this only works because we're passing through a single example
        return score["scores"][0]

    def comet_qe_score(self, prediction: str, source: str) -> float:
        score = self.comet_qe.compute(predictions=[prediction], sources=[source])
        # Single example again
        return score["scores"][0]
