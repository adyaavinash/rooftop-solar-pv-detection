import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

def fetch_satellite_image(lat, lon, zoom=20, size=640):
    """
    Download satellite tile from Google Maps Static API.
    """
    if API_KEY is None:
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}x{size}"
        f"&maptype=satellite&key={API_KEY}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Google API Error: " + response.text)

    return Image.open(BytesIO(response.content))
