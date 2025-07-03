import base64
import io
import requests
import re
from PIL import Image
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def image_url_to_base64(
        image_url : str, 
        max_width : int = 256, 
        max_height : int = 256, 
        quality : int = 85, 
        max_b64_length : int = 65535):
    """
    Fetches an image from a URL, resizes and compresses it, then returns a base64-encoded string.
    Skips encoding if the base64 string would exceed `max_b64_length`.
    """
    try:
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')

        # Open and convert image
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img.thumbnail((max_width, max_height))  # Resize with aspect ratio

        # Compress and save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Create data URI
        base64_string = f"data:{content_type};base64,{base64_image}"

        # Ensure length is within Milvus limits
        if len(base64_string) > max_b64_length:
            logging.debug(f"CATALOG RETRIEVER | utils.image_url_to_base64() | Skipping image: base64 length {len(base64_string)} exceeds limit.")
            return None

        return base64_string

    except requests.RequestException as e:
        logging.debug(f"CATALOG RETRIEVER | utils.image_url_to_base64() | Error fetching image: {e}")
        return None
    except Exception as e:
        logging.debug(f"CATALOG RETRIEVER | utils.image_url_to_base64() | An error occurred: {e}")
        return None

def image_to_base64(image):
    """
    Changes a raw JPEG passed into gradio into the correct format for NVCLIP.
    """
    # Convert the PIL Image to a byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Save the image in JPEG format to the byte stream
    image_bytes = buffered.getvalue()
    
    # Base64 encode the byte stream
    image_b64 = base64.b64encode(image_bytes).decode()
    
    # Return the base64 string in a data URI format
    base64_string = f"data:image/jpeg;base64,{image_b64}"
    
    return base64_string

def is_url(string: str) -> bool:
    """
    Simple check if a string is a URL.
    """
    url_pattern = re.compile(r'^https?://')
    return bool(url_pattern.match(string))