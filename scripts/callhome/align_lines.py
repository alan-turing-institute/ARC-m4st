import sys


def main():
    r"""
    Parse lines to align with the Callhome mapping.

    This is needed, because the line numbers in the mapping files do not align
    with the line numbers in the original .cha files. The approach that this
    script takes is to merge any lines that are not separated by a timestamp,
    which at a glance seems to align the lines correctly.
    """
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} input_file\n")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    prev_line = None
    for line_raw in lines:
        line = line_raw.rstrip("\n")

        # Ignore metadata lines
        if line.startswith("@"):
            continue

        if prev_line is not None:
            # If a line does not end with a timestamp, then merge.
            # Assuming all lines with translation end in a timestamp.
            if not prev_line.endswith("\x15"):
                # Merge current line with the previous one.
                # Split on ': ' to remove the speaker identifier if present.
                parts = line.split(": ", 1)
                if len(parts) > 1:
                    prev_line += " " + parts[1]
                else:
                    prev_line += " " + line
                continue

            # Print previous line if no merge occurred.
            print(prev_line)

        prev_line = line

    if prev_line is not None:
        print(prev_line)


if __name__ == "__main__":
    main()
