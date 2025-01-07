import evaluate
import numpy as np
import pandas as pd
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


class BLASERRefScore:
    """Initialises and applies the BLASER 2.0 QE metric from the SONAR library."""

    def __init__(self) -> None:
        self.blaser_ref = load_blaser_model("blaser_2_0_ref").eval()
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )

    def get_scores(
        self,
        references: pd.Series,
        predictions: pd.Series,
        sources: pd.Series,
        source_lang_codes: pd.Series,
    ) -> float:

        langs = np.unique(source_lang_codes)

        # Store results for all languages so they can be returned together
        results = []

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for language in langs:
            mask = source_lang_codes == language
            sources_lang = np.array(sources[mask])
            refs_lang = np.array(references[mask])
            preds_lang = np.array(predictions[mask])

            src_embs = self.text_embedder.predict(sources_lang, source_lang=language)
            ref_embs = self.text_embedder.predict(refs_lang, source_lang="eng_Latn")
            mt_embs = self.text_embedder.predict(preds_lang, source_lang="eng_Latn")

            for i in range(len(src_embs)):
                result = self.blaser_ref(
                    src=src_embs[[i]], ref=ref_embs[[i]], mt=mt_embs[[i]]
                ).item()
                results.append(result.item())

        return results


class BLASERQEScore:
    """Initialises and applies the BLASER 2.0 reference-based metric from the SONAR
    library."""

    def __init__(self) -> None:
        self.blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )

    def get_scores(
        self, predictions: pd.Series, sources: pd.Series, source_lang_codes: pd.Series
    ) -> float:

        langs = np.unique(source_lang_codes)

        # Store results for all languages so they can be returned together
        results = []

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for language in langs:
            mask = source_lang_codes == language
            sources_lang = np.array(sources[mask])
            preds_lang = np.array(predictions[mask])

            src_embs = self.text_embedder.predict(sources_lang, source_lang=language)
            mt_embs = self.text_embedder.predict(preds_lang, source_lang="eng_Latn")

            for i in range(len(src_embs)):
                result = self.blaser_ref(src=src_embs[[i]], mt=mt_embs[[i]]).item()
                results.append(result.item())

        return results


class COMETRefScore:
    """Applies COMET reference-based metric from the evaluate library."""

    def __init__(self) -> None:

        self.comet = evaluate.load("comet", model="wmt21-comet-mqm")

    def get_scores(
        self, references: pd.Series, predictions: pd.Series, sources: pd.Series
    ) -> float:

        score = self.comet.compute(
            predictions=predictions,
            references=references,
            sources=sources,
        )
        return score["scores"]


class COMETQEScore:
    """Applies COMET QE metric from the evaluate library."""

    def __init__(self) -> None:

        self.comet = evaluate.load("comet", model="wmt21-comet-qe-mqm")

    def comet_qe_score(self, predictions: pd.Series, sources: pd.Series) -> float:
        score = self.comet_qe.compute(predictions=predictions, sources=sources)
        return score["scores"]
