import json

import requests  # type: ignore[import-untyped]


def generate(prompt, context, model_name="llama3.2"):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "context": context,
        },
        stream=False,
    )
    r.raise_for_status()

    response_text = ""
    # Iterating over the returned lines would also allow streaming,
    # but we don't really care about it at the moment.
    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get("response", "")
        response_text += response_part

        if "error" in body:
            raise Exception(body["error"])

        if body.get("done", False):
            break

    return response_text


# NOTE: ollama must be running for this to work, start the ollama
# app or run `ollama serve`
def corrupt_text(text: str, corruption_level: int = 2) -> str:
    r"""
    Get Ollama to corrupt your text to the required 'level'.

    The levels are defined in the prompt to be between 0 and 5, where
    0 is the original text, and 5 is 'very heavy disfluency'. This
    function will modify the prompt with the text and level, and
    return the LLM response.

    This function requires that the Ollama server is running.
    """
    prompt_statement = """
        You will receive a short snippet of text, and your job is to \
corrupt the text, adding disfluencies in such a way, that the text appears \
more conversational. Try to keep the same underlying meaning, and only \
change the style and fluency.

        On the corruption scale 0 to 5, where 0 means return the original \
text without additional disfluency, and 5 means very heavy disfluency, please \
corrupt the text to level {}.

        In your response include only the modified text, and nothing else.

        The text is: {}
    """
    prompt = prompt_statement.format(corruption_level, text)
    return generate(prompt, [])


if __name__ == "__main__":
    # Test
    output = corrupt_text("Hello, I'm Bob.", corruption_level=4)
    print(output)
