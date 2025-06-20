from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, Response # Import Response for sending bytes directly
import shutil, os
# Assuming cartoonizer is in the same directory or accessible via PYTHONPATH
from cartoonizer import cartoonize, upscale_with_realesrgan # Ensure upscale_with_realesrgan is imported
from starlette.middleware.cors import CORSMiddleware # Important for frontend-backend communication
import uuid # For generating unique filenames

app = FastAPI()

# Configure CORS middleware to allow requests from your React frontend
# In development, React typically runs on http://localhost:3000
origins = [
    "http://localhost",
    "http://localhost:3000", # The default port for create-react-app
    "http://127.0.0.1:3000", # Another common local development address
    # You can add more origins here if your frontend is hosted elsewhere (e.g., your deployment URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/cartoonize/")
async def cartoonize_api(
    file: UploadFile = File(...),
    style: str = Form("whitebox") # Accepts 'whitebox', 'sketch', 'oilpaint'
):
    # Generate unique filenames to prevent conflicts if multiple users upload at once
    # and to ensure proper cleanup.
    # Keep original extension for input, but assume .png for intermediate and .jpg for final upscaled.
    original_file_extension = os.path.splitext(file.filename)[1]
    unique_id = str(uuid.uuid4()) # Generate a UUID for uniqueness

    input_filename = f"input_{unique_id}{original_file_extension}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)

    # Intermediate cartoonized image (before upscaling), usually PNG is good for quality
    cartoon_output_filename = f"cartoonized_{style}_{unique_id}.png"
    cartoon_output_path = os.path.join(UPLOAD_DIR, cartoon_output_filename)

    # Final upscaled image, realesrgan saves as JPG by default with -f jpg
    upscaled_output_filename = f"upscaled_{style}_{unique_id}.jpg"
    upscaled_output_path = os.path.join(UPLOAD_DIR, upscaled_output_filename)

    # Variable to hold the image bytes for the response
    response_image_bytes = None

    try:
        # 1. Save the uploaded image temporarily
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved uploaded file to: {input_path}")

        # 2. Apply cartoonization based on the selected style
        # The cartoonize function will handle reading input_path and writing to cartoon_output_path
        cartoonize(input_path, cartoon_output_path, style=style.lower())
        print(f"Cartoonization (style: {style}) saved to: {cartoon_output_path}")

        # 3. Upscale the cartoonized image using Real-ESRGAN
        # upscale_with_realesrgan expects input_path and output_path
        upscale_with_realesrgan(cartoon_output_path, upscaled_output_path)
        print(f"Upscaled image saved to: {upscaled_output_path}")

        # 4. Prepare the final upscaled image for response
        if os.path.exists(upscaled_output_path):
            print(f"Successfully processed. Reading file for response: {upscaled_output_path}")
            # Read the file content into memory
            with open(upscaled_output_path, "rb") as f:
                response_image_bytes = f.read()

            # The file can now be safely removed as its content is in memory
            os.remove(upscaled_output_path)
            print(f"Cleaned up temporary upscaled file: {upscaled_output_path}")

            # Return the image bytes directly
            return Response(content=response_image_bytes, media_type="image/jpeg")
        else:
            print(f"ERROR: Upscaled output file not found at {upscaled_output_path}")
            return {"message": "Processing failed: Upscaled image file was not created."}, 500

    except Exception as e:
        print(f"An unexpected error occurred during image processing: {e}")
        # Return a more descriptive error message to the frontend
        return {"message": f"Image processing failed: {str(e)}"}, 500
    finally:
        # Clean up other temporary files that are not part of the direct response
        for path_to_clean in [input_path, cartoon_output_path]:
            if os.path.exists(path_to_clean):
                try:
                    os.remove(path_to_clean)
                    print(f"Cleaned up temporary file: {path_to_clean}")
                except OSError as e:
                    print(f"Error cleaning up file {path_to_clean}: {e}")




# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import FileResponse
# import shutil, os
# from cartoonizer import cartoonize

# app = FastAPI()
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/cartoonize/")
# async def cartoonize_api(
#     file: UploadFile = File(...),
#     style: str = Form("whitebox")
# ):
#     input_path = os.path.join(UPLOAD_DIR, file.filename)
#     cartoon_output = os.path.join(UPLOAD_DIR, f"{style}_{file.filename}")
#     upscaled_output = os.path.join(UPLOAD_DIR, f"upscaled_{style}_{file.filename}")

#     # Save the uploaded image
#     with open(input_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Apply cartoonization
#     cartoonize(input_path, cartoon_output, style=style.lower())

#     # Upscale the output
#     from cartoonizer import upscale_with_realesrgan
#     upscale_with_realesrgan(cartoon_output, upscaled_output)

#     return FileResponse(upscaled_output)



# # backend/main.py

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import FileResponse
# import shutil, os
# from cartoonizer import cartoonize

# app = FastAPI()
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/cartoonize/")
# async def cartoonize_api(
#     file: UploadFile = File(...),
#     style: str = Form("whitebox")
# ):
#     input_path = os.path.join(UPLOAD_DIR, file.filename)
#     output_path = os.path.join(UPLOAD_DIR, f"{style}_{file.filename}")

#     with open(input_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     cartoonize(input_path, output_path, style=style.lower())
#     return FileResponse(output_path)



# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import FileResponse
# import shutil, os
# from cartoonizer import cartoonize

# app = FastAPI()
# UPLOAD_DIR = "backend/uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/cartoonize/")
# async def cartoonize_api(file: UploadFile = File(...)):
#     input_path = os.path.join(UPLOAD_DIR, file.filename)
#     output_path = os.path.join(UPLOAD_DIR, f"cartoon_{file.filename}")

#     with open(input_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     cartoonize(input_path, output_path)
#     return FileResponse(output_path)
