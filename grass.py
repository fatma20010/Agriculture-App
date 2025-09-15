import cv2
import numpy as np
from skimage import measure
import os
import re
import logging
import asyncio
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from collections import OrderedDict
import tempfile
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from datetime import datetime
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Grass Health Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "Uploads"
IMAGE_OUTPUT_FOLDER = "OutputImages"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_IMAGE_DIMENSION = 1000  # Max width or height in pixels
PROCESSING_TIMEOUT = 20  # Seconds
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_image(img: np.ndarray, filename: str, folder: str) -> str:
    start_time = time.time()
    filepath = os.path.join(folder, filename)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    success = cv2.imwrite(filepath, img, encode_params)
    if not success:
        logger.error(f"Failed to save image at {filepath}")
        raise IOError(f"Failed to save image at {filepath}")
    logger.debug(f"Saved image {filepath} in {time.time() - start_time:.2f} seconds")
    return filepath

async def load_image(image_path: str) -> np.ndarray:
    start_time = time.time()
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}. CWD: {os.getcwd()}")
        raise FileNotFoundError(f"Image not found at {image_path}")
    file_size = os.path.getsize(image_path)
    logger.debug(f"Loading image from {image_path}, size: {file_size} bytes")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"cv2.imread failed to load {image_path}. File may be invalid.")
        raise ValueError(f"Failed to load {image_path}. Size: {file_size} bytes")
    
    # Downscale if too large
    height, width = image.shape[:2]
    if max(height, width) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(height, width)
        new_width, new_height = int(width * scale), int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    logger.debug(f"Loaded image in {time.time() - start_time:.2f} seconds")
    return image

async def segment_grass(image: np.ndarray) -> tuple:
    start_time = time.time()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define green range for grass
    lower_green = np.array([30, 20, 20])
    upper_green = np.array([90, 255, 255])
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
    grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
    # Invert mask to get non-green (bare) areas
    non_green_mask = cv2.bitwise_not(grass_mask)
    logger.debug(f"Segmented grass and non-green areas in {time.time() - start_time:.2f} seconds")
    return grass_mask, non_green_mask

async def classify_health(image: np.ndarray, grass_mask: np.ndarray, non_green_mask: np.ndarray) -> tuple:
    start_time = time.time()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]
    healthy_mask = np.zeros_like(grass_mask)
    unhealthy_mask = np.zeros_like(grass_mask)

    # Calculate healthy and unhealthy percentages
    labels = measure.label(grass_mask, connectivity=2, background=0)
    healthy_pixels = 0
    unhealthy_pixels = 0

    if len(np.unique(labels)) <= 1:  # No grass regions detected
        unhealthy_pixels = np.sum(non_green_mask > 0)
    else:
        for label in np.unique(labels):
            if label == 0:
                continue
            component_mask = (labels == label).astype(np.uint8) * 255
            hsv_component = cv2.bitwise_and(hsv, hsv, mask=component_mask)
            hues = hsv_component[:, :, 0][component_mask > 0]
            saturations = hsv_component[:, :, 1][component_mask > 0]
            mean_hue = np.mean(hues) if hues.size > 0 else 0
            mean_saturation = np.mean(saturations) if saturations.size > 0 else 0

            if 20 <= mean_hue <= 40 and mean_saturation < 100:
                unhealthy_mask |= component_mask
                unhealthy_pixels += np.sum(component_mask > 0)
            else:
                healthy_mask |= component_mask
                healthy_pixels += np.sum(component_mask > 0)
        # Add non-green areas as unhealthy
        unhealthy_mask |= non_green_mask
        unhealthy_pixels += np.sum(non_green_mask > 0)

    healthy_percentage = (healthy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    unhealthy_percentage = (unhealthy_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    logger.debug(f"Classified health in {time.time() - start_time:.2f} seconds")
    return healthy_mask, unhealthy_mask, healthy_percentage, unhealthy_percentage

async def draw_boxes(image: np.ndarray, mask: np.ndarray, color: tuple, output_filename: str, output_folder: str, status: str) -> str:
    start_time = time.time()
    output_image = image.copy()
    overlay = np.zeros_like(image)

    # Apply semi-transparent overlay
    alpha = 0.4
    if np.any(mask):
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
        logger.debug(f"{status} mask applied with {np.sum(mask > 0)} pixels colored {color}")

    await save_image(output_image, output_filename, output_folder)
    logger.debug(f"Drew boxes for {status} in {time.time() - start_time:.2f} seconds")
    return os.path.join(output_folder, output_filename)

async def estimate_compost(image: np.ndarray, unhealthy_mask: np.ndarray) -> float:
    total_pixels = image.shape[0] * image.shape[1]
    unhealthy_pixels = np.sum(unhealthy_mask > 0)
    unhealthy_ratio = unhealthy_pixels / total_pixels
    estimated_total_compost = 1000  # grams
    return unhealthy_ratio * estimated_total_compost

async def process_grass_image(image_path: str, output_folder: str) -> dict:
    start_time = time.time()
    try:
        os.makedirs(output_folder, exist_ok=True)
        input_filename = os.path.basename(image_path)
        safe_filename = re.sub(r'[^\w\-]', '_', os.path.splitext(input_filename)[0])
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        random_suffix = os.urandom(4).hex()

        healthy_output_filename = f"{safe_filename}_healthy_{timestamp}_{random_suffix}.jpg"
        unhealthy_output_filename = f"{safe_filename}_unhealthy_{timestamp}_{random_suffix}.jpg"
        healthy_output_path = os.path.join(output_folder, healthy_output_filename)
        unhealthy_output_path = os.path.join(output_folder, unhealthy_output_filename)
        logger.debug(f"Healthy output will be saved as: {healthy_output_path}")
        logger.debug(f"Unhealthy output will be saved as: {unhealthy_output_path}")

        image = await load_image(image_path)
        grass_mask, non_green_mask = await segment_grass(image)
        healthy_mask, unhealthy_mask, healthy_percentage, unhealthy_percentage = await classify_health(image, grass_mask, non_green_mask)
        compost_needed = await estimate_compost(image, unhealthy_mask)

        # Generate two separate images
        await draw_boxes(image, healthy_mask, (0, 255, 0), healthy_output_filename, output_folder, "Healthy")
        await draw_boxes(image, unhealthy_mask, (0, 0, 255), unhealthy_output_filename, output_folder, "Unhealthy")

        logger.debug(f"Processed image in {time.time() - start_time:.2f} seconds")

        return {
            "healthy_image_path": healthy_output_path,
            "healthy_image_filename": healthy_output_filename,
            "unhealthy_image_path": unhealthy_output_path,
            "unhealthy_image_filename": unhealthy_output_filename,
            "healthy_percentage": float(healthy_percentage),
            "unhealthy_percentage": float(unhealthy_percentage),
            "compost_needed_grams": float(compost_needed)
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "healthy_image_path": "",
            "healthy_image_filename": "",
            "unhealthy_image_path": "",
            "unhealthy_image_filename": "",
            "healthy_percentage": 0.0,
            "unhealthy_percentage": 0.0,
            "compost_needed_grams": 0.0,
            "error": str(e)
        }

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif hasattr(obj, 'item') and callable(obj.item):
        return obj.item()
    else:
        return obj

async def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Deleted file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")

async def clear_output_folder(max_files: int = 50):
    start_time = time.time()
    try:
        files = [f for f in os.listdir(IMAGE_OUTPUT_FOLDER) if os.path.isfile(os.path.join(IMAGE_OUTPUT_FOLDER, f))]
        if len(files) > max_files:
            logger.warning(f"Too many files ({len(files)}) in {IMAGE_OUTPUT_FOLDER}, skipping cleanup")
            return
        for filename in files:
            file_path = os.path.join(IMAGE_OUTPUT_FOLDER, filename)
            await cleanup_file(file_path)
        logger.debug(f"Cleared output folder in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error clearing output folder: {str(e)}")

@app.get("/test-image")
async def generate_test_image():
    start_time = time.time()
    try:
        width, height = 640, 480
        test_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        cv2.putText(test_image, f"Test Image Generated at:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_image, current_time, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        for i in range(10):
            x = np.random.randint(50, width-50)
            y = np.random.randint(200, height-50)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(test_image, (x, y), 30, color, -1)
        filename = f"test_image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        await save_image(test_image, filename, IMAGE_OUTPUT_FOLDER)
        logger.debug(f"Generated test image in {time.time() - start_time:.2f} seconds")
        return FileResponse(os.path.join(IMAGE_OUTPUT_FOLDER, filename), headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Random": str(np.random.random())
        })
    except Exception as e:
        logger.error(f"Error generating test image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating test image: {str(e)}")

@app.get("/")
async def home():
    logger.debug("Received / request")
    return {
        "message": "Welcome to the Grass Health Analysis API",
        "analyze_endpoint": "/analyze_grass",
        "image_endpoint": "/images/{filename}",
        "report_endpoint": "/generate_report",
        "test_endpoint": "/test-image",
        "expected_input": "multipart/form-data with an image file (key: 'image')",
        "allowed_formats": list(ALLOWED_EXTENSIONS)
    }

@app.post("/analyze_grass")
async def analyze_grass(image: UploadFile = File(...), request: Request = None):
    start_time = time.time()
    logger.debug(f"Received /analyze_grass request from {request.client.host if request else 'unknown'} with Content-Type: {image.content_type}")
    file_path = None
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No selected file")
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        await clear_output_folder()
        original_filename = re.sub(r'[^\w\-\.]', '_', image.filename)
        timestamp_suffix = datetime.now().strftime('%Y%m%d%H%M%S%f')
        unique_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp_suffix}{os.path.splitext(original_filename)[1]}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        async with aiofiles.open(file_path, "wb") as f:
            content = await image.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE} bytes")
            img_array = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            await f.write(content)
        logger.debug(f"Saved uploaded file to: {file_path}, size: {len(content)} bytes")

        try:
            results = await asyncio.wait_for(process_grass_image(file_path, IMAGE_OUTPUT_FOLDER), timeout=PROCESSING_TIMEOUT)
            if results is None or "error" in results:
                raise HTTPException(status_code=500, detail="Failed to process image: " + (results.get("error", "Unknown error")))

            # Add analysis_details summary
            healthy = results.get("healthy_percentage", 0.0)
            unhealthy = results.get("unhealthy_percentage", 0.0)
            if healthy >= 80:
                details = "Grass is mostly healthy."
            elif healthy >= 60:
                details = "Grass is moderately healthy with some unhealthy regions."
            elif healthy >= 40:
                details = "Significant unhealthy regions detected."
            else:
                details = "Grass is mostly unhealthy. Immediate action recommended."
            results["analysis_details"] = details

            results = convert_numpy(results)
            logger.debug(f"Processed /analyze_grass in {time.time() - start_time:.2f} seconds")
            response_dict = OrderedDict([
                ("status", "success"),
                ("results", results),
                ("healthy_image_filename", results["healthy_image_filename"]),
                ("unhealthy_image_filename", results["unhealthy_image_filename"])
            ])
            return JSONResponse(content=response_dict, status_code=200, media_type="application/json")
        finally:
            if file_path:
                await cleanup_file(file_path)
    except asyncio.TimeoutError:
        logger.error("Image processing timed out")
        raise HTTPException(status_code=504, detail="Image processing timed out")
    except MemoryError:
        logger.error("Memory error during image processing")
        raise HTTPException(status_code=500, detail="Memory error: Image too large")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path:
            await cleanup_file(file_path)

@app.get("/images/{filename:path}")
async def serve_image(filename: str):
    image_path = os.path.join(IMAGE_OUTPUT_FOLDER, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(image_path, headers=headers)

@app.post("/generate_report")
async def generate_report(image: UploadFile = File(...), request: Request = None):
    start_time = time.time()
    logger.debug(f"Received /generate_report request from {request.client.host if request else 'unknown'} with Content-Type: {image.content_type}")
    pdf_path = None
    file_path = None
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="No selected file")
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        original_filename = re.sub(r'[^\w\-\.]', '_', image.filename)
        timestamp_suffix = datetime.now().strftime('%Y%m%d%H%M%S%f')
        unique_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp_suffix}{os.path.splitext(original_filename)[1]}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        async with aiofiles.open(file_path, "wb") as f:
            content = await image.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE} bytes")
            img_array = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            await f.write(content)
        logger.debug(f"Saved uploaded file to: {file_path}, size: {len(content)} bytes")

        try:
            results = await asyncio.wait_for(process_grass_image(file_path, IMAGE_OUTPUT_FOLDER), timeout=PROCESSING_TIMEOUT)
            if results is None or "error" in results:
                raise HTTPException(status_code=500, detail="Failed to process image: " + (results.get("error", "Unknown error")))

            healthy_percentage = results["healthy_percentage"]
            unhealthy_percentage = results["unhealthy_percentage"]
            compost_needed = results["compost_needed_grams"]
            healthy_image_path = results["healthy_image_path"]
            unhealthy_image_path = results["unhealthy_image_path"]

            # Define timestamp and random_suffix for temp image paths
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            random_suffix = os.urandom(4).hex()

            report_text = (
                f"Healthy Grass Percentage: {healthy_percentage:.1f}%\n"
                f"Unhealthy Grass Percentage: {unhealthy_percentage:.1f}%\n"
            )

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("<b>Grass Health Report</b>", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y, %I:%M %p %Z')}", styles["Normal"]),
                Spacer(1, 12),
            ]

            data = [
                ["Healthy Percentage", f"{healthy_percentage:.1f}%"],
                ["Unhealthy Percentage", f"{unhealthy_percentage:.1f}%"],
            ]
            table = Table(data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]))
            story.extend([
                table,
                Spacer(1, 18),
                Paragraph("<b>Health Assessment</b>", styles["Heading2"]),
                Paragraph(report_text.replace("\n", "<br/>"), styles["Normal"]),
                Spacer(1, 8),
            ])

            recommendation = (
                f"Estimated compost needed: <b>{compost_needed:.2f} grams</b>. "
                f"Apply to enrich soil and promote grass recovery.<br/>"
                f"<b>Metrics:</b><br/>"
                f"- <b>Hue:</b> 20-40 (yellowish, unhealthy), 40-90 (green, healthy).<br/>"
                f"- <b>Saturation:</b> <100 suggests faded grass.<br/>"
                f"Recommendations: Water regularly, improve drainage, aerate soil."
            )
            story.extend([
                Paragraph(recommendation, styles["Normal"]),
                Spacer(1, 18),
                Paragraph("<b>Healthy Regions</b>", styles["Heading2"]),
            ])

            if os.path.exists(healthy_image_path):
                img = cv2.imread(healthy_image_path)
                if max(img.shape[:2]) > 1000:
                    scale = 1000 / max(img.shape[:2])
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    temp_img_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"temp_healthy_{timestamp}_{random_suffix}.jpg")
                    await save_image(img, os.path.basename(temp_img_path), IMAGE_OUTPUT_FOLDER)
                else:
                    temp_img_path = healthy_image_path
                story.extend([
                    Spacer(1, 12),
                    Paragraph("Healthy Regions (Green)", styles["Normal"]),
                    Image(temp_img_path, width=350, height=262.5),
                ])
            else:
                logger.error(f"Healthy image not found at {healthy_image_path}")
                story.extend([
                    Spacer(1, 12),
                    Paragraph("Error including healthy image in report", styles["Normal"]),
                ])

            story.extend([
                Spacer(1, 18),
                Paragraph("<b>Unhealthy Regions</b>", styles["Heading2"]),
            ])

            if os.path.exists(unhealthy_image_path):
                img = cv2.imread(unhealthy_image_path)
                if max(img.shape[:2]) > 1000:
                    scale = 1000 / max(img.shape[:2])
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    temp_img_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"temp_unhealthy_{timestamp}_{random_suffix}.jpg")
                    await save_image(img, os.path.basename(temp_img_path), IMAGE_OUTPUT_FOLDER)
                else:
                    temp_img_path = unhealthy_image_path
                story.extend([
                    Spacer(1, 12),
                    Paragraph("Unhealthy Regions (Red, including non-green/bare areas)", styles["Normal"]),
                    Image(temp_img_path, width=350, height=262.5),
                ])
            else:
                logger.error(f"Unhealthy image not found at {unhealthy_image_path}")
                story.extend([
                    Spacer(1, 12),
                    Paragraph("Error including unhealthy image in report", styles["Normal"]),
                ])

            doc.build(story)
            buffer.seek(0)
            pdf_bytes = buffer.read()
            buffer.close()

            # Create temporary PDF file with explicit error checking
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir()) as temp_pdf:
                try:
                    temp_pdf.write(pdf_bytes)
                    temp_pdf.flush()
                    os.fsync(temp_pdf.fileno())  # Ensure file is written to disk
                    pdf_path = temp_pdf.name
                    if not os.path.exists(pdf_path):
                        logger.error(f"PDF file {pdf_path} was not created")
                        raise IOError(f"PDF file {pdf_path} was not created")
                    logger.debug(f"Created PDF at {pdf_path}")
                except Exception as e:
                    logger.error(f"Error writing PDF file: {str(e)}")
                    raise IOError(f"Error writing PDF file: {str(e)}")

            logger.debug(f"Generated report in {time.time() - start_time:.2f} seconds")
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename="grass_health_report.pdf",
                background=BackgroundTask(cleanup_file, pdf_path)
            )

        except asyncio.TimeoutError:
            logger.error("Report generation timed out")
            raise HTTPException(status_code=504, detail="Report generation timed out")
        except MemoryError:
            logger.error("Memory error during report generation")
            raise HTTPException(status_code=500, detail="Memory error: Image too large")
        except Exception as e:
            logger.error(f"Error generating report content: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing report request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path:
            await cleanup_file(file_path)

@app.get("/direct-image/{filename:path}")
async def direct_image_access(filename: str):
    image_path = os.path.join(IMAGE_OUTPUT_FOLDER, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Disposition": f"inline; filename={filename}",
                "X-Random": f"{datetime.now().timestamp()}"
            }
        )
    except Exception as e:
        logger.error(f"Error accessing image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accessing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with 1 worker...")
    # Listen on all interfaces for frontend compatibility
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")