# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import binascii
import io
import requests
import re
from PIL import Image, UnidentifiedImageError
import logging
import sys
import os


def normalize_runtime_image_input(image_data: str) -> str:
    """Validate runtime image input and return a base64 data URI.

    Runtime queries may provide either a base64 image data URI or raw base64.
    URLs and filesystem paths are intentionally rejected so untrusted requests
    cannot trigger server-side network or file access.
    """
    value = image_data.strip()
    if not value:
        raise ValueError("image_base64 must contain base64 image data")

    if value.startswith("data:"):
        try:
            header, payload = value.split(",", 1)
        except ValueError as exc:
            raise ValueError("image_base64 must be a base64 image data URI") from exc

        media_type = header[5:].split(";", 1)[0]
        if not header.endswith(";base64") or not (
            media_type.startswith("image/") or media_type == "application/octet-stream"
        ):
            raise ValueError("image_base64 must be a base64 image data URI")
    else:
        payload = value

    compact_payload = "".join(payload.split())
    if not compact_payload:
        raise ValueError("image_base64 must contain base64 image data")

    padded_payload = compact_payload + "=" * (-len(compact_payload) % 4)
    image_format = None
    try:
        image_bytes = base64.b64decode(padded_payload, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            image.verify()
    except (binascii.Error, UnidentifiedImageError, ValueError, OSError) as exc:
        raise ValueError("image_base64 must contain valid base64 image data") from exc

    if value.startswith("data:"):
        return f"{header},{padded_payload}"
    if not image_format:
        raise ValueError("image_base64 must contain valid base64 image data")
    return f"data:image/{image_format.lower()};base64,{padded_payload}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def image_path_to_base64(
        image_path: str,
        max_width : int = 256, 
        max_height : int = 256, 
        quality : int = 85, 
        max_b64_length : int = 65535) -> str:
    """
    Converts an image to a base64 string.
    """
    shared_root = os.environ.get("SHARED_ROOT", "/app/shared")
    with open(os.path.join(shared_root, image_path.lstrip("/")), "rb") as image_file:
        img = Image.open(image_file).convert("RGB")
        img.thumbnail((max_width, max_height))  # Resize with aspect ratio

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        base64_string = f"data:image/jpeg;base64,{base64_image}"

        # Ensure length is within Milvus limits
        if len(base64_string) > max_b64_length:
            logging.debug(f"CATALOG RETRIEVER | utils.image_url_to_base64() | Skipping image: base64 length {len(base64_string)} exceeds limit.")
            return None

        return base64_string

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

def is_path(string: str) -> bool:
    """
    Simple check if a string is a path.
    """
    path_pattern = re.compile(r'^/')
    return bool(path_pattern.match(string))

def resize_base64_image(base64_string: str, max_width: int = 256, max_height: int = 256, quality: int = 85, max_b64_length: int = 65535) -> str:
    """
    Resize a base64 image to fit within the size limits.
    """
    try:
        # Extract base64 data from data URI
        if base64_string.startswith('data:'):
            header, base64_data = base64_string.split(',', 1)
        else:
            base64_data = base64_string
            header = 'data:image/jpeg;base64'
        
        # Decode and resize
        image_data = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img.thumbnail((max_width, max_height))
        
        # Re-encode
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        resized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"{header},{resized_base64}"
    except Exception as e:
        logging.error(f"Error resizing image: {e}")
        return None
