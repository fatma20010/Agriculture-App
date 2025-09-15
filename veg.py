import cv2
import numpy as np
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import OrderedDict as OrderedDictType
from collections import OrderedDict
import os
import logging
import tempfile
import shutil
from together import Together
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from io import BytesIO
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Authentication settings

app = FastAPI()

# Custom CORS middleware
class CORSMiddlewareCustom(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add CORS headers to every response
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
        response.headers["Access-Control-Allow-Headers"] = "*, Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization"
        response.headers["Access-Control-Expose-Headers"] = "*"
        
        return response

# Add custom CORS middleware
app.add_middleware(CORSMiddlewareCustom)

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=86400,        # Cache preflight requests for 24 hours
)

# Add OPTIONS route handler for preflight requests
@app.options("/{path:path}")
async def options_handler(request: Request, path: str):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Authentication API routes
# Configuration
UPLOAD_FOLDER = 'Uploads'
IMAGE_OUTPUT_FOLDER = 'OutputImages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mv2'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

# Together AI API Key for report generation
TOGETHER_API_KEY = "5a5d3ff7a2fbae72418501e22ced7935f285982c800882c7ba03e2e44e999025"
client = Together(api_key=TOGETHER_API_KEY)

# Initialize model variables
model = None
base_model = None
class_names = {}

# Try to load TensorFlow and the model
try:
    import tensorflow as tf
    logger.debug("Loading model...")
    if os.path.exists('plant_species_model.h5'):
        try:
            # Try direct import first
            model = tf.keras.models.load_model('plant_species_model.h5')
        except AttributeError:
            # If tf.keras is not available, try importing keras directly
            import keras
            model = keras.models.load_model('plant_species_model.h5')
        
        # Dummy pass to build model
        logger.debug("Performing dummy prediction...")
        dummy_img = np.zeros((1, 224, 224, 3))
        _ = model.predict(dummy_img)
        
        # Define intermediate model to extract CNN features
        logger.debug("Creating intermediate model...")
        try:
            base_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-6].output)
        except AttributeError:
            # If tf.keras is not available, use keras directly
            import keras
            base_model = keras.models.Model(inputs=model.input, outputs=model.layers[-6].output)
    else:
        logger.warning("Model file 'plant_species_model.h5' not found. API will run in limited mode.")
except (ImportError, AttributeError) as e:
    logger.error(f"Failed to import TensorFlow or load model: {str(e)}")
    logger.warning("API will run in limited mode without model functionality")

# Load class indices if available
try:
    logger.debug("Loading class indices...")
    if os.path.exists('class_indices.json'):
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
    else:
        logger.warning("Class indices file 'class_indices.json' not found.")
except Exception as e:
    logger.error(f"Failed to load class indices: {str(e)}")

# Image settings
IMAGE_SIZE = (224, 224)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_image(img, filename: str, folder: str) -> str:
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, img)
    return filepath

def extract_features(image_path: str) -> OrderedDictType:
    logger.debug(f"Processing image: {image_path}")
    # Check if model is loaded
    if model is None or base_model is None:
        logger.error("Model not loaded. Cannot process image.")
        return OrderedDict([
            ('error', 'Model not loaded'),
            ('status', 'error'),
            ('message', 'The server is running in limited mode without the ML model.')
        ])
        
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        raise ValueError("Failed to load image")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # 1. Yellowing detection (HSV)
    logger.debug("Detecting yellowing...")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    yellow_percentage = np.count_nonzero(mask_yellow) / mask_yellow.size * 100

    # 2. Spot detection
    logger.debug("Detecting spots...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = len(contours)

    # 3. Feature map extraction
    logger.debug("Extracting feature maps...")
    feature_maps = base_model.predict(img_array)
    feature_map = np.mean(feature_maps, axis=(0, 1, 2)) if len(feature_maps.shape) == 4 else feature_maps[0]

    # 4. Prediction
    logger.debug("Making prediction...")
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    predicted_species = class_names.get(predicted_class, f"Unknown-{predicted_class}")

    # 5. Compost estimation (rule-based)
    logger.debug("Estimating compost...")
    alpha = 0.5  # g per %_abi5b95d8fe
    beta = 0.1   # g per spot
    estimated_compost = (yellow_percentage * alpha) + (spot_count * beta)

    # Save processed images
    filename_base = os.path.basename(image_path).rsplit('.', 1)[0]
    original_path = save_image(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), f"{filename_base}_original.jpg", IMAGE_OUTPUT_FOLDER)
    mask_path = save_image(mask_yellow, f"{filename_base}_yellowing_mask.jpg", IMAGE_OUTPUT_FOLDER)
    edges_path = save_image(edges, f"{filename_base}_edges.jpg", IMAGE_OUTPUT_FOLDER)

    # Return results with image paths
    return OrderedDict([
        ('species', predicted_species),
        ('confidence', float(confidence)),
        ('yellow_percentage', float(yellow_percentage)),
        ('spot_count', int(spot_count)),
        ('estimated_compost_grams', float(estimated_compost)),
        ('original_image_path', original_path),
        ('yellowing_mask_path', mask_path),
        ('edges_image_path', edges_path)
    ])

def escape_latex(s):
    """Escape LaTeX special characters in a string."""
    if not isinstance(s, str):
        s = str(s)
    return (
        s.replace('\\', r'\\textbackslash ')
         .replace('&', r'\&')
         .replace('%', r'\%')
         .replace('$', r'\$')
         .replace('#', r'\#')
         .replace('_', r'\_')
         .replace('{', r'\{')
         .replace('}', r'\}')
         .replace('~', r'\textasciitilde ')
         .replace('^', r'\textasciicircum ')
         .replace('"', r'"')
    )

@app.get('/health')
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        'status': 'healthy',
        'message': 'Plant Species Detection API is running',
        'version': '1.0.0'
    }

@app.get('/')
async def home():
    logger.debug("Received / request")
    return {
        'message': 'Welcome to the Plant Species Detection API',
        'endpoint': '/predict',
        'image_endpoint': '/images/{filename}',
        'report_endpoint': '/generate_report',
        'expected_input': 'multipart/form-data with an image file (key: "image")',
        'allowed_formats': list(ALLOWED_EXTENSIONS)
    }

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    logger.debug(f"Received /predict request")
    if not image:
        raise HTTPException(status_code=400, detail='No image provided')
    
    if image.filename is None:
        raise HTTPException(status_code=400, detail='No filename provided')
        
    if not allowed_file(image.filename):
        raise HTTPException(
            status_code=400,
            detail=f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        )

    # Save the uploaded file
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        results = extract_features(file_path)
        if 'error' in results:
            return JSONResponse(content=results, status_code=500)
            
        response_dict = OrderedDict([
            ('status', 'success'),
            ('results', results)
        ])
        return JSONResponse(content=response_dict, status_code=200)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get('/images/{filename:path}')
@app.head('/images/{filename:path}')
async def serve_image(filename: str):
    # This endpoint is intentionally NOT protected by authentication
    # to allow images to be loaded by the frontend directly
    
    # Remove the OutputImages prefix if present
    if filename.startswith('OutputImages/'):
        filename = filename.replace('OutputImages/', '')
    
    image_path = os.path.join(IMAGE_OUTPUT_FOLDER, filename)
    logger.debug(f"Serving image: {image_path}")
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail='Image not found')

@app.post('/generate_report')
async def generate_report(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    logger.debug(f"Received /generate_report request")
    pdf_path = None
    if not image:
        raise HTTPException(status_code=400, detail='No image provided')
    
    if image.filename is None:
        raise HTTPException(status_code=400, detail='No filename provided')
        
    if not allowed_file(image.filename):
        raise HTTPException(
            status_code=400,
            detail=f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        )

    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        results = extract_features(file_path)
        if 'error' in results:
            raise HTTPException(status_code=500, detail=results['message'])
            
        species = results['species']
        confidence = results['confidence']
        yellow_percentage = results['yellow_percentage']
        spot_count = results['spot_count']
        original_path = results['original_image_path']
        mask_path = results['yellowing_mask_path']
        edges_path = results['edges_image_path']
        estimated_compost = results['estimated_compost_grams']

        # Generate report text using Together AI LLM
        prompt = f"""
        Generate a concise plant health report in English based on the following analysis:
        - Species: {species}
        - Confidence: {confidence:.2f}
        - Yellowing Percentage: {yellow_percentage:.2f}%
        - Spot Count: {spot_count}
        Explain if the plant is unhealthy based on the yellowing and spot count. For {species}, provide a simple explanation of the disease (e.g., Late Blight for Tomato___Late_blight), including what it is and its symptoms. Compare natural spots (typically small, uniform, and green) with unhealthy spots (irregular, brown, or numerous), and suggest how the current spot count and yellowing indicate health. Keep the tone professional yet accessible. Do NOT mention compost or nutrition recommendations in your answer."""
        
        # Default report text in case of API failure
        report_text = f"""
        Based on the analysis of your {species} plant:
        
        Yellowing: {yellow_percentage:.2f}% of the plant shows yellowing, which {'may indicate stress or nutrient deficiency' if yellow_percentage > 5 else 'is within normal parameters'}.
        
        Spot Count: {spot_count} spots detected. {'This high number of spots may indicate a potential disease issue.' if spot_count > 10 else 'The number of spots is within normal range for a healthy plant.'} 
        
        Overall Assessment: {'Your plant shows signs of potential health issues that should be monitored.' if (yellow_percentage > 5 or spot_count > 10) else 'Your plant appears to be in good health based on visual indicators.'}
        
        Recommendation: {'Consider inspecting the plant more closely for signs of disease or infestation.' if (yellow_percentage > 5 or spot_count > 10) else 'Continue with regular care and monitoring.'}
        """
        
        try:
            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[
                    {"role": "system", "content": "You are a plant health expert tasked with generating concise and informative plant health reports."},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            
            # Extract content safely - avoiding attribute access that causes linter errors
            try:
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                        content = response.choices[0].message.content
                        if content:
                            report_text = content
            except AttributeError:
                # Try dictionary-style access if attribute access fails
                if response and isinstance(response, dict) and 'choices' in response:
                    if response['choices'] and isinstance(response['choices'][0], dict) and 'message' in response['choices'][0]:
                        if 'content' in response['choices'][0]['message']:
                            content = response['choices'][0]['message']['content']
                            if content:
                                report_text = content
        except Exception as e:
            logger.error(f"Error generating report with LLM: {str(e)}")
            # We already have a fallback report_text, so no need to set it again

        # Generate PDF with ReportLab
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("<b>Plant Health Report</b>", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Date:</b> June 23, 2025", styles['Normal']))
        story.append(Spacer(1, 12))

        # Table of results
        data = [
            ["Species", species],
            ["Confidence", f"{confidence:.2f}"],
            ["Yellowing Percentage", f"{yellow_percentage:.2f}%"],
            ["Spot Count", str(spot_count)]
        ]
        table = Table(data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ]))
        story.append(table)
        story.append(Spacer(1, 18))
        story.append(Paragraph("<b>Health Assessment and Disease Information</b>", styles['Heading2']))
        
        # Handle report text safely
        if isinstance(report_text, str):
            story.append(Paragraph(report_text.replace('\n', '<br/>'), styles['Normal']))
        else:
            story.append(Paragraph("Unable to generate report text.", styles['Normal']))
            
        story.append(Spacer(1, 8))

        # Add compost recommendation and additional recommendations
        compost_recommendation = (
            f"Estimated compost: <b>{estimated_compost:.2f} grams</b>. You can use this compost to enrich your soil, improve plant health, and support sustainable gardening. Composting is an environmentally friendly way to recycle plant material and boost future crop yields."
        )
        additional_recommendations = (
            "Maintain regular monitoring for disease symptoms, ensure proper plant spacing for air circulation, and use clean tools to prevent the spread of pathogens. Water plants at the base to avoid wetting foliage and remove any severely affected leaves to reduce disease pressure."
        )
        story.append(Paragraph(f"<b>Recommendation:</b> {compost_recommendation}<br/>{additional_recommendations}", styles['Normal']))
        story.append(Spacer(1, 18))
        story.append(Paragraph("<b>Images</b>", styles['Heading2']))
        for img_path, caption in zip([original_path, mask_path, edges_path], ["Original Image", "Yellowing Mask", "Edges (Spots)"]):
            if os.path.exists(img_path):
                story.append(Spacer(1, 12))
                story.append(Paragraph(caption, styles['Normal']))
                story.append(Image(img_path, width=200, height=200))
        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.read()
        buffer.close()

        # Save to a temp file for send_file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_bytes)
            pdf_path = temp_pdf.name

        background_tasks.add_task(os.remove, pdf_path)
        return FileResponse(
            pdf_path,
            media_type='application/pdf',
            filename=f"{species}_report.pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        # Do NOT remove pdf_path here

@app.get('/grass-images/{filename:path}')
@app.head('/grass-images/{filename:path}')
async def serve_grass_image(filename: str):
    try:
        import requests
        
        # Remove the OutputImages prefix if present
        clean_filename = filename
        if 'OutputImages/' in clean_filename:
            clean_filename = clean_filename.split('OutputImages/')[-1]
            
        logger.debug(f"Forwarding image request: {clean_filename}")
        
        # Forward the request to the grass API
        response = requests.get(f'http://localhost:5000/images/{clean_filename}', stream=True)
        
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type=response.headers.get('content-type', 'image/jpeg')
            )
        else:
            logger.error(f"Image not found in grass server. Status: {response.status_code}, URL: {clean_filename}")
            raise HTTPException(status_code=response.status_code, detail='Image not found in grass server')
    except Exception as e:
        logger.error(f"Error forwarding to grass image API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/analyze_grass")
async def analyze_grass(request: Request, image: UploadFile = File(...)):
    logger.debug(f"RECEIVED REQUEST at /analyze_grass with content-type: {request.headers.get('content-type')}")
    # Forward the request to the grass API
    try:
        # Save the uploaded file
        filename = image.filename or "uploaded_image.jpg"  # Provide a default filename if None
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # Create a new form with the saved file
        import requests
        from requests_toolbelt.multipart.encoder import MultipartEncoder
        
        with open(file_path, "rb") as f:
            form = MultipartEncoder(fields={
                'image': (filename, f, 'image/jpeg')
            })
            
            headers = {
                'Content-Type': form.content_type
            }
            
            # Forward to grass API
            response = requests.post(
                'http://localhost:5000/analyze_grass',
                headers=headers,
                data=form
            )
            
        # Clean up file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Parse response
        if response.status_code == 200:
            response_data = response.json()
            logger.debug(f"Raw response data: {response_data}")
            
            if 'results' in response_data:
                results = response_data['results']
                
                # Ensure required fields are present and map them
                mapped_results = {
                    'healthy_percentage': results.get('healthy_percentage', 0.0),
                    'unhealthy_percentage': results.get('unhealthy_percentage', 0.0),
                    'compost_needed_grams': results.get('compost_needed_grams', 0.0),
                    'analysis_details': results.get('analysis_details', ''),
                    'healthy_image_path': results.get('healthy_image_filename', ''),
                    'unhealthy_image_path': results.get('unhealthy_image_filename', '')
                }
                # Optionally, include any other fields you want to pass through
                response_data['results'] = mapped_results
                logger.debug(f"Processed grass analysis results: {mapped_results}")
        
        # Return the response
        return JSONResponse(
            content=response_data,
            status_code=response.status_code
        )
        
    except Exception as e:
        logger.error(f"Error forwarding to grass API: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )
@app.post('/generate_grass_report')
async def generate_grass_report(image: UploadFile = File(...)):
    logger.debug(f"Received /generate_grass_report request")
    try:
        # Save the uploaded file
        filename = image.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        # Create a new form with the saved file
        import requests
        from requests_toolbelt.multipart.encoder import MultipartEncoder
        
        with open(file_path, "rb") as f:
            form = MultipartEncoder(fields={
                'image': (filename, f, 'image/jpeg')
            })
            
            headers = {
                'Content-Type': form.content_type,
                'Accept': 'application/pdf'
            }
            
            # Forward to grass API
            response = requests.post(
                'http://localhost:5000/generate_report',
                headers=headers,
                data=form,
                stream=True
            )
            
        # Clean up file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Return the PDF response
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type=response.headers.get('content-type', 'application/pdf'),
                headers={
                    'Content-Disposition': 'attachment; filename="grass_health_report.pdf"'
                }
            )
        else:
            logger.error(f"Grass API returned status {response.status_code}: {response.text}")
            return JSONResponse(
                content={"status": "error", "message": "Failed to generate grass report"},
                status_code=response.status_code
            )
            
    except Exception as e:
        logger.error(f"Error forwarding to grass report API: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )
    results = []
    for record in records:
        try:
            # Parse the JSON data - convert SQLAlchemy Column to string first
            data = json.loads(str(record.result_data))
            
            # Create a summary with just the essential information
            summary = {
                "species": data.get("species", "Unknown"),
                "confidence": data.get("confidence", 0),
                "yellow_percentage": data.get("yellow_percentage", 0),
                "spot_count": data.get("spot_count", 0),
                "estimated_compost_grams": data.get("estimated_compost_grams", 0)
            }
            
            results.append({
                "id": record.id,
                "analysis_type": record.analysis_type,
                "result_summary": summary,
                "created_at": record.created_at
            })
        except Exception as e:
            logger.error(f"Error processing record {record.id}: {str(e)}")
    
   

@app.get('/api/images/{filepath:path}')
@app.head('/api/images/{filepath:path}')
async def serve_api_image(filepath: str):
    """
    Proxy endpoint to match frontend URL structure: /api/images/...
    This endpoint is intentionally NOT protected by authentication
    """
    # Remove any OutputImages prefix if present
    if filepath.startswith('OutputImages/'):
        filepath = filepath.replace('OutputImages/', '')
    
    image_path = os.path.join(IMAGE_OUTPUT_FOLDER, filepath)
    logger.debug(f"Serving image from API path: {image_path}")
    
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail='Image not found')

if __name__ == '__main__':
    logger.info("Starting Farm Plant Analysis Server on port 5002...")
    uvicorn.run(app, host='0.0.0.0', port=5002)