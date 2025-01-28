import os
import uuid
from collections.abc import Iterator

from pydub import AudioSegment
from tqdm import tqdm

from m4st.parse import TranscriptParser


class CallhomePipeline:
    def __init__(
        self,
        text_folder: str,
        audio_folder: str | None,
        translation_model,
        transcription_model,
        audio_snippet_dir: str | None = None,
    ):
        """
        Initialise the pipeline.

        It is assumed that corresponding text and audio files have unique
        identifiers XXXX.cha and XXXX.mp3.

        Args:
            text_folder (str): Path to the folder containing .cha files.
            audio_folder (str): Path to the folder containing .mp3 files.
            translation_model: Instance of the TranslationModel.
            transcription_model: Instance of the TranscriptionModel.
            audio_snippet_dir (Optional[str]): Directory to save audio snippets.
                If None, snippets are not saved.
        """
        self.parser = TranscriptParser.from_folder(text_folder)
        self.audio_folder = audio_folder
        self.translation_model = translation_model
        self.transcription_model = transcription_model
        self.audio_snippet_dir = audio_snippet_dir

        if audio_snippet_dir and not os.path.exists(audio_snippet_dir):
            os.makedirs(audio_snippet_dir)

    def get_audio_snippet(self, audio_file: str, timestamp: tuple) -> AudioSegment:
        """
        Args:
            audio_file (str): Path to the audio file.
            timestamp (tuple): A tuple of start and end times, as given by
                the .cha file.

        Returns:
            AudioSegment
        """
        start, end = timestamp
        all_sound = AudioSegment.from_file(audio_file)

        return all_sound[start:end]

    def __iter__(self) -> Iterator[dict]:
        """
        Create an iterator that yields datapoints.

        Yields:
            dict: A dictionary containing the original text snippet, translated text,
                  transcribed audio, and optionally the saved audio snippet file path.
        """
        for line_idx in tqdm(range(len(self.parser.lines))):
            timestamp = self.parser.timestamps[line_idx]
            # Only record datapoints with a timestamp, so that we can
            # have an audio record of them (for BLASER).
            if timestamp is None:
                continue

            snippet = self.parser.lines[line_idx]
            # Turn into tuple of ints
            timestamp = (int(timestamp.split("_")[0]), int(timestamp.split("_")[1]))
            file_prefix = self.parser.line_files[line_idx]

            output_dict = {
                "text_snippet": snippet,
                "timestamp": timestamp,
                "file_prefix": file_prefix,
            }

            # Translate the text snippet
            if self.translation_model is not None:
                translated_text = self.translation_model(snippet)
                output_dict["translated_text"] = translated_text

            if self.audio_folder is not None:
                audio_file = os.path.join(self.audio_folder, f"{file_prefix}.mp3")
                # Extract the audio snippet
                audio_snippet = self.get_audio_snippet(audio_file, timestamp)

                if self.transcription_model is not None:
                    transcribed_audio = self.transcription_model(audio_snippet)
                    output_dict["transcribed_audio"] = transcribed_audio

                if self.audio_snippet_dir:
                    # Generate random unique file prefix
                    file_id = uuid.uuid4().hex
                    audio_snippet_path = os.path.join(
                        self.audio_snippet_dir, f"{file_id}.mp3"
                    )
                    audio_snippet.export(audio_snippet_path, format="mp3").close()
                    output_dict["audio_snippet_id"] = f"{file_id}.mp3"

            yield output_dict


if __name__ == "__main__":
    # Some test code
    text_folder = (
        "/Users/bvodenicharski/repos/ARC-m4st/experiments/callhome/data/eng/text"
    )
    audio_folder = (
        "/Users/bvodenicharski/repos/ARC-m4st/experiments/callhome/data/eng/audio"
    )

    translation_model = None
    transcription_model = None
    audio_snippet_dir = "./audio_test_out"

    pipe = CallhomePipeline(
        text_folder,
        audio_folder,
        translation_model,
        transcription_model,
        audio_snippet_dir=audio_snippet_dir,
    )

    pipe_iter = pipe.__iter__()
    print(next(pipe_iter))
    print(next(pipe_iter))
