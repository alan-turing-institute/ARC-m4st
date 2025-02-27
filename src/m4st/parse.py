import glob
import os
import re

from tqdm import tqdm


class TranscriptParser:
    r"""
    Provides a bag of conversational lines.

    Instantiate this by using the `from_folder` class method and
    pointing it to a folder from the CallHome dataset, for example
    the 'deu' folder for transcriptions in German. This class will
    try its best to remove the .cha format specifics, and only
    keep the UTF8 characters, thus providing text we can use for
    downstream translation.
    """

    def __init__(self):
        self.lines = []
        self.timestamps = []
        # Keep track from which file prefix the line has come.
        # This makes downstream processing easier.
        self.line_files = []

    @classmethod
    def from_folder(cls, folder_path: str):
        parser = cls()
        # Loop through all .cha files in the folder
        for file_path in tqdm(
            glob.glob(os.path.join(folder_path, "*.cha")), desc=f"Parsing {folder_path}"
        ):
            with open(file_path) as file:
                data = file.read()
                parser.parse_transcription(data, file_prefix=file_path[-8:-4])

        return parser

    @classmethod
    def from_file(cls, file_path: str):
        parser = cls()
        with open(file_path) as file:
            data = file.read()
            parser.parse_transcription(data, file_prefix=file_path[-8:-4])

        return parser

    def parse_line(self, line: str, file_prefix=None):
        # Match lines with participant utterances
        match = re.match(r"\*(\w):\s+(.*)", line)
        if match:
            participant, text = match.groups()
            # Remove timestamps (e.g., •50770_51060•) from the text
            # And other artefacts

            timestamp = re.search(r"\x15\d+_\d+\x15", text)

            clean_text = re.sub(r"\x15\d+_\d+\x15", "", text).strip()
            clean_text = re.sub(r"&=\S+", "", clean_text).strip()
            clean_text = re.sub(r"&+\S+", "", clean_text).strip()
            clean_text = re.sub(r"\+/", "", clean_text).strip()
            clean_text = re.sub(r"\+", "", clean_text).strip()
            if clean_text in [".", "?", "!"]:
                # Nothing but the punctuation is remaining
                return

            self.lines.append(clean_text)
            self.line_files.append(file_prefix)

            # Occasionally, the line will not have a timestamp, so
            # handle this case without regex complaining.
            if timestamp is not None:
                self.timestamps.append(timestamp.group()[1:-1])
            else:
                self.timestamps.append(None)

    def parse_transcription(self, data: str, file_prefix=None):
        lines = data.split("\n")
        for line in lines:
            if line in ["@Begin", "@UTF8", "@End"]:
                # The begin header
                pass
            elif line.startswith("*"):
                # Participant line
                self.parse_line(line, file_prefix=file_prefix)
