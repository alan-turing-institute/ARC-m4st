from abc import ABC, abstractmethod
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class TranscriptionModel(ABC):
    @abstractmethod
    def __call__(self, path_to_file: Path) -> str:
        r"""
        Transcribe the speech from an audio file.
        """


class WhisperTranscription(TranscriptionModel):
    def __init__(self, language: str):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.language = language

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def __call__(self, audio_file: Path) -> str:
        return self.pipe(str(audio_file), generate_kwargs={"language": self.language})[
            "text"
        ]


if __name__ == "__main__":
    file_path = Path("e46f7c492d9f46509e001a55faf30f08.mp3")
    whisper = WhisperTranscription("English")
    print(whisper(file_path))
