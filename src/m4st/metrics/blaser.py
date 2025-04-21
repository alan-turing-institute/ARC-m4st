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

        unique_lang_pairs = sorted(
            {
                (src, tgt)
                for src, tgt in zip(source_languages, target_languages, strict=False)
            }
        )  # unique language pairs sorted by source language then target language

        # Store results for all languages so they can be returned together
        results = np.full(len(dataset), np.nan, dtype=float)

        # at best we can batch by language pair as the embedder predict methods require
        # a single language to be specified for all inputs
        for src_lang, tgt_lang in tqdm(unique_lang_pairs):
            mask = (source_languages == src_lang) & (target_languages == tgt_lang)
            sources_lang = np.array(dataset.source)[mask]
            preds_lang = np.array(dataset.prediction)[mask]

            # get unique sources to avoid re-embedding the same source multiple times.
            # source_idx_lang gives the mapping from the original sources to the unique
            # sources. The same strategy is used for the predictions and references.
            # This can give a large speedup when duplicates are present, e.g. if many
            # translation systems are being evaluated on the same sources.
            unique_sources_lang, source_idx_lang = np.unique(
                sources_lang, return_inverse=True
            )

            # embed inputs
            embeds = {}
            if self.audio_source:
                speech_embedder = self.get_speech_embedder(src_lang)
                # load and resample audio
                unique_audio_sources = []
                for src in unique_sources_lang:
                    waveform, sr = torchaudio.load(src)
                    if sr != BLASER_SAMPLE_RATE:
                        waveform = resample(waveform, sr, BLASER_SAMPLE_RATE)
                    unique_audio_sources.append(waveform)

                unique_speech_embeds = speech_embedder.predict(
                    unique_audio_sources, progress_bar=True
                )  # shape: (n_unique_sources, embed_dim)
                embeds["src"] = unique_speech_embeds[
                    source_idx_lang
                ]  # shape: (n_sources, embed_dim)
            else:
                unique_text_embeds = self.text_embedder.predict(
                    unique_sources_lang, source_lang=src_lang, progress_bar=True
                )  # shape: (n_unique_sources, embed_dim)

                embeds["src"] = unique_text_embeds[
                    source_idx_lang
                ]  # shape: (n_sources, embed_dim)

            unique_preds_lang, preds_idx_lang = np.unique(
                preds_lang, return_inverse=True
            )
            unique_preds_embeds = self.text_embedder.predict(
                unique_preds_lang, source_lang=tgt_lang, progress_bar=True
            )  # shape: (n_unique_preds, embed_dim)
            embeds["mt"] = unique_preds_embeds[
                preds_idx_lang
            ]  # shape: (n_preds, embed_dim)

            if not self.qe:
                refs_lang = np.array(dataset.reference)[mask]
                # get unique references to avoid re-embedding the same reference
                unique_refs_lang, ref_idx_lang = np.unique(
                    refs_lang, return_inverse=True
                )
                unique_refs_embeds = self.text_embedder.predict(
                    unique_refs_lang, source_lang=tgt_lang, progress_bar=True
                )  # shape: (n_unique_refs, embed_dim)
                embeds["ref"] = unique_refs_embeds[
                    ref_idx_lang
                ]  # shape: (n_refs, embed_dim)

            # dict of lists to list of dicts
            inputs = [
                {k: v[[i]] for k, v in embeds.items()}
                for i in range(len(embeds["src"]))
            ]

            lang_results = [self.blaser(**sample).item() for sample in tqdm(inputs)]
            results[mask] = lang_results

        return list(results)
