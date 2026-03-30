import requests
import torch
import os
import io
import requests

temperature = 0
max_completion_tokens = 2048
server_url = os.environ.get('server_url')

class CustomModel():
    def __init__(self, model_path, language_only):
        self.model_path = model_path
        self.language_only = language_only
        self.model_type = 'custom'
        

    def respond(self, prompt, obs=None):
        data = {"sentence": prompt}

        if isinstance(obs, (list, tuple)):
            file_handles = []
            files = []
            try:
                for image_path in obs:
                    img_file = open(image_path, "rb")
                    file_handles.append(img_file)
                    files.append(("image", (os.path.basename(image_path), img_file, "image/png")))
                response = requests.post(server_url, files=files, data=data)
            finally:
                for img_file in file_handles:
                    img_file.close()
        elif obs is not None:
            with open(obs, "rb") as img_file:
                files = {"image": img_file}
                response = requests.post(server_url, files=files, data=data)
        else:
            response = requests.post(server_url, data=data)

        res= response.json()['response']
        if response.status_code != 200:
            print("Error:", response.text)
        return res
