import io

from PIL import Image


def compress_image_in_memory(image_bytes: bytes, quality: int = 80) -> bytes:
    # Create a file-like buffer for the input bytes
    input_buffer = io.BytesIO(image_bytes)

    with Image.open(input_buffer) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Create an output buffer to hold the compressed image
        output_buffer = io.BytesIO()

        # Save to the buffer instead of a file path
        img.save(output_buffer, format="JPEG", optimize=True, quality=quality)

        # Return the raw compressed bytes
        return output_buffer.getvalue()
