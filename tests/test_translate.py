import pytest

from m4st.translate.model import OllamaTranslateModel, T5TranslateModel

eng_text = "Hi, my name is Bob."


def test_t5():
    t5 = T5TranslateModel("eng", "fra")
    translated_text_t5 = t5(eng_text)
    print(translated_text_t5)


@pytest.mark.skip(
    reason="Ollama server must be set up and running, please test this locally."
)
def test_llama32():
    llama32translate = OllamaTranslateModel("eng", "spa")
    translated_text_llama32 = llama32translate(eng_text)
    print(translated_text_llama32)
