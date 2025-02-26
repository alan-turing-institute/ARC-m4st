import sys


def main():
    r""" 
    Removes the delimiter encoding which person is speaking.
    """
    # Use stdin for reading the input
    for line in sys.stdin:
        # Skip the person identifier (e.g. A, B, B1, etc.)
        clean_line = " ".join(line.split(" ")[1:]).strip()

        print(clean_line)


if __name__ == "__main__":
    main()
