import numpy as np
import torchaudio
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from torchaudio.functional import resample
from tqdm import tqdm

from m4st.metrics import Metric, TranslationDataset

BLASER_SAMPLE_RATE = 16000


class BLASERScore(Metric):
    """Initialises and applies the BLASER 2.0 QE metric from the SONAR library."""

    def __init__(self, qe: bool = False, audio_source: bool = False) -> None:
        self.name = "blaser_2_0"
        self.data_req_inputs = [
            "prediction",
            "source",
            "source_language",
            "target_language",
        ]
        self.qe = qe
        if self.qe:
            self.name += "_qe"
        else:
            self.name += "_ref"
            self.data_req_inputs.append("reference")

        self.blaser = load_blaser_model(self.name).eval()
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
        )

        self.audio_source = audio_source
        # self._speech_embedder_language and self._speech_embedder are set when
        # get_scores is called and an audio_source is requested. We don't pre-load it
        # here is the speech embedder to load depends on the language in the dataset
        # to be scored.
        self._speech_embedder_language = ""

    def get_speech_embedder(self, source_lang: str) -> SpeechToEmbeddingModelPipeline:
        """Get the speech embedder for the given source language, immediately returning
        the previously cached one, if available.
        """
        if self._speech_embedder_language != source_lang:
            # assumes source_lang is either a valid speech encoder code, e.g. "eng", or
            # a text encode code, e.g. "eng_Latn"
            encoder_name = f"sonar_speech_encoder_{source_lang.split('_')[0]}"
            self._speech_embedder = SpeechToEmbeddingModelPipeline(encoder=encoder_name)
            self._speech_embedder_language = source_lang

        return self._speech_embedder

    def get_scores(self, dataset: TranslationDataset) -> list[float]:
        self.check_dataset_compatible(dataset, audio_source=self.audio_source)

        source_languages = np.array(dataset.source_language)
        target_languages = np.array(dataset.target_language)

        unique_lang_pairs = {
            (src, tgt)
            for src, tgt in zip(source_languages, target_languages, strict=False)
        }

        # Store results for all languages so they can be returned together
        results = np.full(len(dataset), np.nan, dtype=float)

        # BLASER requires the source language, so at best we can batch by language as
        # source_lang must be a string
        for src_lang, tgt_lang in tqdm(unique_lang_pairs):
            mask = (source_languages == src_lang) & (target_languages == tgt_lang)
            sources_lang = np.array(dataset.source)[mask]
            preds_lang = np.array(dataset.prediction)[mask]

            # embed inputs
            embeds = {}
            if self.audio_source:
                speech_embedder = self.get_speech_embedder(src_lang)
                # load and resample audio
                audio_sources = []
                for src in sources_lang:
                    waveform, sr = torchaudio.load(src)
                    if sr != BLASER_SAMPLE_RATE:
                        waveform = resample(waveform, sr, BLASER_SAMPLE_RATE)
                    audio_sources.append(waveform)

                embeds["src"] = speech_embedder.predict(audio_sources)
            else:
                embeds["src"] = self.text_embedder.predict(
                    sources_lang, source_lang=src_lang
                )

            embeds["mt"] = self.text_embedder.predict(preds_lang, source_lang=tgt_lang)

            if not self.qe:
                refs_lang = np.array(dataset.reference)[mask]
                embeds["ref"] = self.text_embedder.predict(
                    refs_lang, source_lang=tgt_lang
                )

            # dict of lists to list of dicts
            inputs = [
                {k: v[[i]] for k, v in embeds.items()}
                for i in range(len(embeds["src"]))
            ]

            lang_results = [self.blaser(**sample).item() for sample in tqdm(inputs)]
            results[mask] = lang_results

        return list(results)
