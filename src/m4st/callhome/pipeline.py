import os
import uuid
from collections.abc import Iterator

from pydub import AudioSegment
from tqdm import tqdm

from m4st.callhome.process_mappings import process


class CallhomePipeline:
    def __init__(
        self,
        text_folder: str,
        eng_folder: str,
        mapping_folder: str,
        audio_folder: str | None,
        translation_model,
        transcription_model,
        audio_snippet_dir: str | None = None,
        split: str = "devtest",
    ):
        """
        Initialise the pipeline.

        It is assumed that corresponding text and audio files have unique
        identifiers XXXX.cha and XXXX.mp3. The English translation files
        are assumed to be one of 'callhome_{devtest | evltest | train}.en',
        and the mapping files of the same name, but without the '.en' suffix.

        Args:
            text_folder (str): Path to the folder containing .cha files.
            eng_folder (str): Path to the folder containing .en files.
            mapping_folder (str): Path to the folder containing mapping files.
            audio_folder (str): Path to the folder containing .mp3 files.
            translation_model: Instance of the TranslationModel.
            transcription_model: Instance of the TranscriptionModel.
            audio_snippet_dir (Optional[str]): Directory to save audio snippets.
                If None, snippets are not saved.
        """
        # Folders
        # self.text_folder = text_folder
        # self.eng_folder = eng_folder
        # self.mapping_folder = mapping_folder
        self.audio_folder = audio_folder
        self.audio_snippet_dir = audio_snippet_dir

        # Models
        self.translation_model = translation_model
        self.transcription_model = transcription_model

        assert split in ["devtest", "evltest", "train"]
        self.spa_eng_df = process(
            text_folder,
            os.path.join(mapping_folder, f"callhome_{split}"),
            os.path.join(eng_folder, f"callhome_{split}.en"),
        )

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
        # NOTE Iterating through a DataFrame is not ideal, but doing it anyway for
        # ease, since the neural networks will be the expensive operation, rather
        # than df iteration.
        for line_idx in tqdm(range(len(self.spa_eng_df))):
            row = self.spa_eng_df.iloc[line_idx]
            timestamp = row["timestamp"]

            assert (
                timestamp is not None
            ), "If this error occurs, something has gone wrong during pre-processing \
                Callhome."

            snippet = row["spa"]
            eng_translation = row["eng"]
            # Turn into tuple of ints
            timestamp = (int(timestamp.split("_")[0]), int(timestamp.split("_")[1]))
            file_prefix = row["prefix"].split(".")[0]

            output_dict = {
                "original": snippet,
                "reference_english": eng_translation,
                "audio_timestamp": timestamp,
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
    text_folder = "/Users/bvodenicharski/repos/ARC-m4st/data/spa_processed"
    audio_folder = None  # (
    #    "/Users/bvodenicharski/repos/ARC-m4st/experiments/callhome/data/eng/audio"
    # )
    eng_folder = (
        "/Users/bvodenicharski/repos/ARC-m4st/data/fisher_ch_spa-eng/data/corpus/ldc"
    )
    mapping_folder = (
        "/Users/bvodenicharski/repos/ARC-m4st/data/fisher_ch_spa-eng/data/mapping"
    )

    translation_model = None
    transcription_model = None
    audio_snippet_dir = None  # "./audio_test_out"

    pipe = CallhomePipeline(
        text_folder,
        eng_folder,
        mapping_folder,
        audio_folder,
        translation_model,
        transcription_model,
        audio_snippet_dir=audio_snippet_dir,
    )

    pipe_iter = pipe.__iter__()
    print(next(pipe_iter))
    print(next(pipe_iter))
