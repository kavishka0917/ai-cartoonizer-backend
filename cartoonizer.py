import os
import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from huggingface_hub import snapshot_download
from subprocess import run

# Load White-box Cartoonizer from HuggingFace
whitebox_model_dir = snapshot_download("sayakpaul/whitebox-cartoonizer")
whitebox_model = tf.saved_model.load(whitebox_model_dir)
whitebox_infer = whitebox_model.signatures["serving_default"]

def preprocess_whitebox(img: np.ndarray) -> tf.Tensor:
    h, w, _ = img.shape
    scale = 720 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    nh8, nw8 = nh // 8 * 8, nw // 8 * 8
    img_cropped = img_resized[:nh8, :nw8]
    img_norm = img_cropped.astype(np.float32) / 127.5 - 1
    return tf.reshape(img_norm, (1, nh8, nw8, 3))

def apply_whitebox(img_path, out_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_whitebox(rgb)
    output = whitebox_infer(input_tensor)["final_output:0"].numpy()[0]
    output = ((output + 1) * 127.5).astype(np.uint8)
    result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, result)

def apply_sketch(img_path, out_path):
    img = cv2.imread(img_path, 0)  # grayscale
    inv = 255 - img
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(img, 255 - blur, scale=256)
    cv2.imwrite(out_path, sketch)

def apply_oilpaint(img_path, out_path):
    img = cv2.imread(img_path)
    # Downscale for speed
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Apply repeated bilateral filter
    for _ in range(5):
        small = cv2.bilateralFilter(small, d=9, sigmaColor=75, sigmaSpace=75)

    # Resize back to original size
    filtered = cv2.resize(small, (img.shape[1], img.shape[0]))

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Invert and convert to 3 channels
    edges_inv = cv2.bitwise_not(edges)
    edges_inv = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # Combine edge mask and oil filtered image
    oil_paint_result = cv2.bitwise_and(filtered, edges_inv)

    cv2.imwrite(out_path, oil_paint_result)

def apply_animegan_style(img_path, out_path, sess, input_name):
    original = cv2.imread(img_path)
    original_h, original_w = original.shape[:2]

    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (256, 256))
    normed = resized.astype(np.float32) / 127.5 - 1
    input_tensor = normed[np.newaxis, :, :, :]

    output = sess.run(None, {input_name: input_tensor})[0][0]
    output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    final = cv2.resize(output_bgr, (original_w, original_h))
    cv2.imwrite(out_path, final)

def upscale_with_realesrgan(input_path, output_path):
    # Assumes realesrgan-ncnn-vulkan binary is present and executable
    binary = "./models/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan"
    model_dir = "./models/realesrgan-ncnn-vulkan/models"

    run([binary,
         "-i", input_path,
         "-o", output_path,
         "-n", "realesrgan-x4plus",
         "-s", "4",
         "-f", "jpg",
         "-m", model_dir])

def cartoonize(input_path, output_path, style="whitebox"):
    if style == "whitebox":
        apply_whitebox(input_path, output_path)
    elif style == "sketch":
        apply_sketch(input_path, output_path)
    elif style == "oilpaint":
        apply_oilpaint(input_path, output_path)
    else:
        raise ValueError(f"Unknown style: {style}")



# # backend/cartoonizer.py

# import cv2
# import numpy as np
# import tensorflow as tf
# from huggingface_hub import snapshot_download
# from PIL import Image
# import os

# # Download model files once
# MODEL_DIR = snapshot_download("sayakpaul/whitebox-cartoonizer")
# cartoon_model = tf.saved_model.load(MODEL_DIR)
# infer = cartoon_model.signatures["serving_default"]

# def preprocess(img: np.ndarray) -> tf.Tensor:
#     h, w, _ = img.shape
#     scale = 720 / max(h, w)
#     nh, nw = int(h * scale), int(w * scale)
#     img_resized = cv2.resize(img, (nw, nh))
#     # pad to multiple of 8
#     nh8, nw8 = nh // 8 * 8, nw // 8 * 8
#     img_cropped = img_resized[:nh8, :nw8]
#     img_norm = img_cropped.astype(np.float32) / 127.5 - 1
#     return tf.reshape(img_norm, (1, nh8, nw8, 3))

# def cartoonize(input_path: str, output_path: str) -> str:
#     img = cv2.imread(input_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     inp = preprocess(img_rgb)
#     output = infer(inp)["final_output:0"].numpy()[0]
#     output = ((output + 1) * 127.5).astype(np.uint8)
#     out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(output_path, out_bgr)
#     return output_path
