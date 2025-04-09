import time
import uuid

import requests


class AzureTranslateModel:
    def __init__(self):
        # Add your key and endpoint
        self.key = "<YOUR_ENDPOINT_KEY>"
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
    model = AzureTranslateModel()

    for i in range(331):
        with open(f"data/source_merged/merged_source_{i}.txt") as f:
            source = f.read()

        # translation = ". ".join(model(s) for s in source.split("."))
        translation = model(source)

        with open(f"data/translation_merged/merged_translation_{i}.txt", "w") as f:
            f.write(translation)

        time.sleep(3)
