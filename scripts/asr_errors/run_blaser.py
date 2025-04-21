"""
Score a source and translation group with BLASER-2 QE, with both audio and text sources.
"""

from utils import (
    get_audio_path,
    get_group,
    get_group_hypotheses,
    get_group_sources,
    get_scores_path,
)

from m4st.metrics import TranslationDataset
from m4st.metrics.blaser import BLASERScore

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
    hypotheses = get_group_hypotheses(group)

    from_lang = "eng_Latn"
    to_lang = "spa_Latn"

    print("Running BLASER audio...")
    blaser = BLASERScore(qe=True, audio_source=True)
    audio_source_path = get_audio_path()
    audio_scores = blaser.get_scores(
        TranslationDataset(
            prediction=hypotheses,
            source=[audio_source_path] * len(hypotheses),
            source_language=[from_lang] * len(hypotheses),
            target_language=[to_lang] * len(hypotheses),
        )
    )
    audio_scores_path = get_scores_path(group, "blaser_audio")
    with open(audio_scores_path, "w") as f:
        for score in audio_scores:
            f.write(f"{score}\n")

    print("Running BLASER text...")
    blaser = BLASERScore(qe=True, audio_source=False)
    text_sources = get_group_sources(group)
    text_scores = blaser.get_scores(
        TranslationDataset(
            prediction=hypotheses,
            source=text_sources,
            source_language=[from_lang] * len(hypotheses),
            target_language=[to_lang] * len(hypotheses),
        )
    )
    text_scores_path = get_scores_path(group, "blaser_text")
    with open(text_scores_path, "w") as f:
        for score in text_scores:
            f.write(f"{score}\n")


if __name__ == "__main__":
    main()
