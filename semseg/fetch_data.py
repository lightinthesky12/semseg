import requests
from PIL import Image
import io
import numpy as np


def load_image_from_url(url):
    """Loads an image from a URL, and loads into a numpy array"""
    try:
        response = requests.get(url, verify=False)
        img = Image.open(io.BytesIO(response.content))
        img_data = np.array(img)
        return img_data
    except Exception as e:
        print("Fetching image at " + url + " failed: " + str(e))
        raise e

