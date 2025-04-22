"""
Create a sequence of source with a varying number of words replaced with similar words
(according to Llama 3.2)
"""

import re

from ollama import chat
from utils import get_group_src_path, get_original_source

if __name__ == "__main__":
    source = get_original_source()
    output_path = get_group_src_path("perturbed")

    words = re.split(r"(\s|,|\.)", source)
    perturbed_words = []
    for w in words:
        if w in [" ", "", ",", "."]:  # preserve spaces and punctuation
            perturbed_words.append(w)
        else:
            similar_word = "one two"
            while len(similar_word.split()) > 1:
                response = chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f'What is a similar word to "{w}"? '
                                "Reply with only a single word."
                            ),
                        },
                    ],
                )
                similar_word = response.message.content.strip()

            similar_word = re.sub(r'[\.?!"]', "", similar_word)
            if w[0].isupper():
                similar_word = similar_word.capitalize()
            else:
                similar_word = similar_word.lower()
            perturbed_words.append(similar_word)
            print(w, "-->", similar_word)

    print("".join(perturbed_words))

    for i in range(len(words)):
        perturbed_source = "".join(words[:i] + perturbed_words[i:])
        with open(f"{output_path}_{i}.txt", "w") as f:
            f.write(perturbed_source)
