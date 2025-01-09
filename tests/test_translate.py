from m4st.translate.model import OllamaTranslateModel, T5TranslateModel

eng_text = "Hi, my name is Bob."


def test_t5():
    t5 = T5TranslateModel()
    translated_text_t5 = t5(eng_text, source_lang_iso="eng", target_lang_iso="fra")
    print(translated_text_t5)


def test_llama32():
    llama32translate = OllamaTranslateModel()
    translated_text_llama32 = llama32translate(
        eng_text, source_lang_iso="eng", target_lang_iso="fra"
    )
    print(translated_text_llama32)
