"""
Use Azure Translate API to translate a directory of modified source files into Spanish.
"""

import os
import time
import uuid

import requests
from utils import get_group, get_group_mt_path, get_group_sources


class AzureTranslateModel:
    def __init__(self):
        # Add your key and endpoint
        try:
            self.key = os.environ["AZURE_TRANSLATE_KEY"]
        except KeyError as e:
            msg = "Please set the AZURE_TRANSLATE_KEY environment variable."
            raise KeyError(msg) from e
        endpoint = "https://api.cognitive.microsofttranslator.com"

        # location, also known as region.
        # required if you're using a multi-service or regional (not global) resource.
        # It can be found in the Azure portal on the Keys and Endpoint page.
        self.location = "uksouth"

        path = "/translate"
        self.constructed_url = endpoint + path

        self.params = {"api-version": "3.0", "from": "en", "to": ["es"]}

    def __call__(self, text: str):
        # You can pass more than one object in body.
        body = [{"text": text}]
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            # location required if you're using a multi-service or regional
            # (not global) resource.
            "Ocp-Apim-Subscription-Region": self.location,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }
        request = requests.post(
            self.constructed_url, params=self.params, headers=headers, json=body
        )
        response = request.json()
        print("---")
        print(response)
        return response[0]["translations"][0]["text"]


if __name__ == "__main__":
    group = get_group()
    output_path = get_group_mt_path(group)
    sources = get_group_sources(group)

    model = AzureTranslateModel()
    for i, source in enumerate(sources):
        translation = model(source)

        with open(f"{output_path}_{i}.txt", "w") as f:
            f.write(translation)

        time.sleep(3)  # Avoid hitting the rate limit
