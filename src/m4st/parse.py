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

    @classmethod
    def from_folder(cls, folder_path: str):
        parser = cls()
        # Loop through all .cha files in the folder
        for file_path in tqdm(
            glob.glob(os.path.join(folder_path, "*.cha")), desc=f"Parsing {folder_path}"
        ):
            with open(file_path) as file:
                data = file.read()
                parser.parse_transcription(data)

        return parser

    def parse_line(self, line: str):
        # Match lines with participant utterances
        match = re.match(r"\*(\w):\s+(.*)", line)
        if match:
            participant, text = match.groups()
            # Remove timestamps (e.g., •50770_51060•) from the text
            # And other artefacts
            clean_text = re.sub(r"\x15\d+_\d+\x15", "", text).strip()
            clean_text = re.sub(r"&=\S+", "", clean_text).strip()
            clean_text = re.sub(r"&+\S+", "", clean_text).strip()
            clean_text = re.sub(r"\+/", "", clean_text).strip()
            clean_text = re.sub(r"\+", "", clean_text).strip()
            if clean_text in [".", "?", "!"]:
                # Nothing but the punctuation is remaining
                return

            self.lines.append(clean_text)

    def parse_transcription(self, data: str):
        lines = data.split("\n")
        for line in lines:
            if line in ["@Begin", "@UTF8", "@End"]:
                # The begin header
                pass
            elif line.startswith("*"):
                # Participant line
                self.parse_line(line)
