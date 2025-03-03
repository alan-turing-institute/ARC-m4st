import re
import sys


def main():
    r"""
    Cleans a stream of transcript text.

    Use by piping in the text of a transcript file, and either
    save the output to disk, or redirect to the next script.
    """
    timestamp_pattern = r"\x15\d+_\d+\x15"
    # Use stdin for reading the input
    for line in sys.stdin:
        clean_line = re.sub(timestamp_pattern, "", line).strip()
        clean_line = re.sub(r"&=\S+", "", clean_line).strip()
        clean_line = re.sub(r"&+\S+", "", clean_line).strip()
        clean_line = re.sub(r"\+/", "", clean_line).strip()
        clean_line = re.sub(r"\+", "", clean_line).strip()

        print(clean_line)


if __name__ == "__main__":
    main()
