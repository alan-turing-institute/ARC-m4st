"""
Score a source and translation group with BLASER-2 QE, with both audio and text sources.
"""

import torchaudio.functional as F
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from utils import (
    get_group,
    get_group_hypotheses,
    get_group_sources,
    get_reference,
    get_scores_path,
    get_source_audio,
)

map_lang = {
    "zh": "zho_Hans",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "hi": "hin_Deva",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "en": "eng_Latn",
    "uk": "ukr_Cyrl",
    "is": "isl_Latn",
}


def main() -> None:
    group = get_group()
    text_sources = get_group_sources(group)
    hypotheses = get_group_hypotheses(group)
    reference = get_reference()
    text_scores_path = get_scores_path(group, "blaser_text")

    audio_source, sample_rate = get_source_audio()
    audio_source = F.resample(audio_source, sample_rate, 16000)
    audio_scores_path = get_scores_path(group, "blaser_audio")

    blaser = load_blaser_model("blaser_2_0_qe").eval()

    from_lang = "eng_Latn"
    to_lang = "spa_Latn"
    speech_encoder = "sonar_speech_encoder_eng"
    text_encoder = "text_sonar_basic_encoder"

    print("Setting up model pipelines...")
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder=speech_encoder)
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder=text_encoder, tokenizer=text_encoder
    )

    # Get embeddings for source text, references, and source audio
    # These will be common across translation models
    print("Getting audio source embeddings...")
    audio_src_emb = s2vec_model.predict([audio_source])
    print("Getting reference embeddings...")
    ref_emb = t2vec_model.predict([reference], source_lang=to_lang)
    print("Getting text source embeddings...")
    text_src_emb = t2vec_model.predict([text_sources[-1]], source_lang=from_lang)

    result_text = blaser(src=text_src_emb, mt=ref_emb).item()
    print("REF TEXT:", result_text)
    result_audio = blaser(src=audio_src_emb, mt=ref_emb).item()
    print("REF AUDIO:", result_audio)

    text_src_embeds = t2vec_model.predict(text_sources, source_lang=from_lang)
    hyp_embeds = t2vec_model.predict(hypotheses, source_lang=to_lang)
    text_scores = blaser(src=text_src_embeds, mt=hyp_embeds)
    text_scores = text_scores.squeeze().tolist()
    with open(text_scores_path, "w") as f:
        for score in text_scores:
            f.write(f"{score}\n")

    audio_src_embeds = audio_src_emb.repeat(hyp_embeds.shape[0], 1)
    audio_scores = blaser(src=audio_src_embeds, mt=hyp_embeds)
    audio_scores = audio_scores.squeeze().tolist()
    with open(audio_scores_path, "w") as f:
        for score in audio_scores:
            f.write(f"{score}\n")


if __name__ == "__main__":
    main()
