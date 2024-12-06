# Metrics for Speech Translation (M4ST)

[![Actions Status][actions-badge]][actions-link]

Evaluation of metrics for Speech Translation

## Installation

From source:
```bash
git clone https://github.com/alan-turing-institute/ARC-M4ST
cd ARC-M4ST
python -m pip install .
```

## Usage

# Compiling notes

`brew install typst`

`typst compile notes.typ`

## CallHome Dataset

Go [https://ca.talkbank.org/access/CallHome](here), select the conversation language, create account, then you can download the "media folder". There you can find the .cha files, which contain the transcriptions.

To load the transcriptions as a bag of sentences, use `m4st.parse.TranscriptParser.from_folder` to load all conversation lines. This class does not group them by participant, or conversation - it just loads every lines as an entry to a list (+ some pre-processing).


## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-M4ST/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-M4ST/actions
<!-- prettier-ignore-end -->
