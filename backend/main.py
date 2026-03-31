from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pseudo-3D Pipeline API")

# Configure CORS for Next.js frontend (default port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Pseudo-3D Pipeline API"}

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    logger.info(f"Received file: {image.filename} with type {image.content_type}")
    
    # Read file content to get size, then reset read pointer
    content = await image.read()
    size_mb = len(content) / (1024 * 1024)
    await image.seek(0)

    logger.info(f"File size: {size_mb:.2f} MB")

    # Day 1-2 placeholder response
    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "size_mb": round(size_mb, 2),
        "status": "success",
        "message": "Image uploaded successfully. Depth processing and segmentation will be implemented."
    }
