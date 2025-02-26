from abc import ABC, abstractmethod

from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from m4st.ollama.client import generate

language_iso_to_name = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "jpn": "Japanese",
    "spa": "Spanish",
    "rou": "Romanian",
    "zho": "Chinese",
}


class TranslationModel(ABC):
    @abstractmethod
    def __call__(self, text: str):
        pass


class NLLBTranslateModel(TranslationModel):
    def __init__(
        self,
        source_lang_iso: str,
        target_lang_iso: str,
        model_tag: str = "facebook/nllb-200-distilled-600M",
    ):

        self.language_iso_to_nllb = {
                "eng" : "eng_Latn",
                "spa" : "spa_Latn"
        }

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_tag)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)

        self.supported_languages: list[str] = ["eng", "spa"]
        assert target_lang_iso in self.supported_languages, "NLLB \
                Actually supports 200 languages, but this model only \
                expects to work on Spanish and English."
        assert source_lang_iso in self.supported_languages, "NLLB \
                Actually supports 200 languages, but this model only \
                expects to work on Spanish and English."

        self.source_lang_iso: str = source_lang_iso
        self.target_lang_iso: str = target_lang_iso

    def __call__(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        # NOTE There used to be a max_length=30 parameter. I'm removing it, but
        # keep in mind this is a sentence-level translation model.
        # TODO Do you explicitly set the input language?
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.language_iso_to_nllb[self.target_lang_iso]))

        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


class T5TranslateModel(TranslationModel):
    def __init__(
        self,
        source_lang_iso: str,
        target_lang_iso: str,
    ):
        model_tag = "google-t5/t5-small"
        self.model = T5ForConditionalGeneration.from_pretrained(model_tag)
        self.tokenizer = T5Tokenizer.from_pretrained(model_tag)

        self.supported_languages: list[str] = ["eng", "deu", "fra"]

        if target_lang_iso not in self.supported_languages:
            err_msg = f"This model only supports {self.supported_languages}, \
but got target {target_lang_iso}."
            raise Exception(err_msg)
        if source_lang_iso not in self.supported_languages:
            err_msg = f"This model only supports {self.supported_languages}, \
but got source {source_lang_iso}."
            raise Exception(err_msg)

        self.source_lang_iso: str = source_lang_iso
        self.target_lang_iso: str = target_lang_iso

    def __call__(self, text: str):
        source_lang_name, target_lang_name = (
            language_iso_to_name[self.source_lang_iso],
            language_iso_to_name[self.target_lang_iso],
        )
        model_input_text = f"translate {source_lang_name} to {target_lang_name}: {text}"
        model_input_tokens = self.tokenizer(
            model_input_text, return_tensors="pt"
        ).input_ids
        model_output_tokens = self.model.generate(
            model_input_tokens, max_new_tokens=210
        )

        return self.tokenizer.decode(model_output_tokens[0], skip_special_tokens=True)


class OllamaTranslateModel(TranslationModel):
    def __init__(
        self, source_lang_iso: str, target_lang_iso: str, model_tag: str = "llama3.2"
    ):
        self.model_tag = model_tag

        self.source_lang_iso: str = source_lang_iso
        self.target_lang_iso: str = target_lang_iso

    def __call__(self, text: str):
        source_lang_name, target_lang_name = (
            language_iso_to_name[self.source_lang_iso],
            language_iso_to_name[self.target_lang_iso],
        )
        prompt = f"""
            Please translate the following sentence from {source_lang_name}\
 to {target_lang_name}, and return only the translated sentence as your\
 response: {text}
        """

        return generate(prompt, [], model_name=self.model_tag)
