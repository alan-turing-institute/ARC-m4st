import re
import sys


def main():
    r""" """
    timestamp_pattern = r"\x15\d+_\d+\x15"
    # Use stdin for reading the input
    for line in sys.stdin:
        timestamp = re.search(timestamp_pattern, line)
        print(timestamp.group()[1:-1])


if __name__ == "__main__":
    main()
