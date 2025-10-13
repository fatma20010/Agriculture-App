# Terra Verra - Plant Health Analysis System

A comprehensive web application for analyzing plant health using computer vision and machine learning. This system helps users identify plant species, detect diseases, assess grass health, and receive actionable recommendations for plant care.

## 🌟 Features

### Plant Species Detection
- **ML-Powered Classification**: Uses a trained TensorFlow/Keras model to identify plant species
- **Disease Detection**: Analyzes yellowing, spots, and other visual indicators of plant health
- **Confidence Scoring**: Provides confidence levels for predictions
- **Compost Recommendations**: Estimates compost needs based on plant condition

### Grass Health Analysis
- **Health Segmentation**: Distinguishes between healthy and unhealthy grass areas
- **Visual Analysis**: Generates color-coded overlay images showing health status
- **Detailed Metrics**: Provides percentage breakdowns of healthy vs. unhealthy areas
- **Actionable Insights**: Offers recommendations for grass care and improvement

### Report Generation
- **PDF Reports**: Professional reports with analysis results and images
- **Visual Documentation**: Includes original, processed, and annotated images
- **AI-Generated Insights**: Uses LLM (Mixtral-8x7B) for detailed health assessments
- **Downloadable**: Easy-to-share PDF format for record-keeping

### User Management
- **Authentication**: Secure JWT-based authentication system
- **User Profiles**: Individual user accounts with personalized data
- **Analysis History**: Track and review past analyses
- **Role-Based Access**: Admin and regular user roles

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **TailwindCSS** for styling
- **shadcn/ui** component library
- **React Router** for navigation
- **Axios** for API communication
- **TanStack Query** for data fetching

### Backend
- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - Machine learning model
- **OpenCV** - Image processing
- **SQLAlchemy** - Database ORM
- **SQLite** - Database
- **JWT** - Authentication
- **ReportLab** - PDF generation
- **Together AI** - LLM integration for report generation

### Image Processing
- **OpenCV (cv2)** - Image manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-image** - Advanced image processing

## 📋 Prerequisites

- **Node.js** (v16 or higher)
- **Python** (v3.8 or higher)
- **pip** (Python package manager)
- **npm** or **bun** (Node package manager)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Inveepplant
```

### 2. Frontend Setup
```bash
# Install dependencies
npm install
# or
bun install
```

### 3. Backend Setup
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install required packages
pip install fastapi uvicorn python-multipart opencv-python numpy scikit-image
pip install tensorflow keras reportlab passlib python-jose sqlalchemy aiofiles
pip install together requests requests-toolbelt
```

### 4. Model Files
Ensure the following files are present in the root directory:
- `plant_species_model.h5` - Trained plant species classification model
- `class_indices.json` - Class label mappings for the model

## 🏃 Running the Application

### Option 1: Using the Batch File (Windows)
```bash
run_servers.bat
```

### Option 2: Using Python Script
```bash
python start_servers.py
```

### Option 3: Manual Start

**Terminal 1 - Grass Health API:**
```bash
python grass.py
```

**Terminal 2 - Plant Species API:**
```bash
python veg.py
```

**Terminal 3 - Frontend:**
```bash
npm run dev
# or
bun run dev
```

## 🌐 API Endpoints

### Grass Health Analysis API (Port 5000)
- `GET /` - API information
- `POST /analyze_grass` - Analyze grass health from image
- `POST /generate_report` - Generate PDF report for grass analysis
- `GET /images/{filename}` - Serve processed images
- `GET /test-image` - Generate test image

### Plant Species Detection API (Port 5002)
- `GET /` - API information
- `POST /predict` - Predict plant species and health
- `POST /generate_report` - Generate PDF report for plant analysis
- `GET /images/{filename}` - Serve processed images
- `POST /analyze_grass` - Proxy to grass analysis API
- `GET /health` - Health check endpoint

### Authentication Endpoints
- `POST /register` - Create new user account
- `POST /login` - User authentication
- `GET /users/me` - Get current user information

## 📁 Project Structure

```
Inveepplant/
├── src/                      # Frontend source files
│   ├── components/           # React components
│   │   └── ui/              # shadcn/ui components
│   ├── pages/               # Page components
│   ├── hooks/               # Custom React hooks
│   └── lib/                 # Utility functions
├── public/                   # Static assets
├── Uploads/                  # Temporary upload directory
├── OutputImages/            # Processed images
├── grass.py                 # Grass health analysis API
├── veg.py                   # Plant species detection API
├── vegup.py                 # Alternative plant API
├── models_and_deps.py       # Database models and dependencies
├── user_history_api.py      # User history management
├── auth_example.py          # Authentication examples
├── proxy_server.py          # API proxy server
├── plant_species_model.h5   # ML model file
├── class_indices.json       # Class mappings
├── users.db                 # SQLite database
├── package.json             # Frontend dependencies
├── vite.config.ts           # Vite configuration
├── tailwind.config.ts       # TailwindCSS configuration
└── tsconfig.json            # TypeScript configuration
```

## 🔧 Configuration

### Backend Configuration
Edit configuration variables in the respective Python files:

**grass.py:**
- `UPLOAD_FOLDER` - Upload directory path
- `IMAGE_OUTPUT_FOLDER` - Output directory path
- `MAX_FILE_SIZE` - Maximum upload size (default: 16MB)
- `MAX_IMAGE_DIMENSION` - Maximum image dimension (default: 1000px)
- `PROCESSING_TIMEOUT` - Processing timeout (default: 20s)

**veg.py:**
- `TOGETHER_API_KEY` - Together AI API key for LLM integration
- `IMAGE_SIZE` - Input image size for model (default: 224x224)

**models_and_deps.py:**
- `SECRET_KEY` - JWT secret key (change in production!)
- `ALGORITHM` - JWT algorithm (default: HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES` - Token expiration time

### Frontend Configuration
Edit `vite.config.ts` for proxy settings and build configurations.

## 📊 Features in Detail

### Image Analysis Process

1. **Upload**: User uploads plant or grass image
2. **Preprocessing**: Image is resized and normalized
3. **Analysis**:
   - Plant species prediction
   - Yellowing detection using HSV color space
   - Spot detection using edge detection
   - Health classification
4. **Results**: Visual overlays and detailed metrics
5. **Report**: AI-generated recommendations and PDF export

### Health Metrics

**Plant Analysis:**
- Species identification with confidence score
- Yellowing percentage (HSV-based)
- Spot count (Canny edge detection)
- Estimated compost needs

**Grass Analysis:**
- Healthy grass percentage
- Unhealthy/bare area percentage
- Color-coded visualization
- Compost recommendations

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- File type validation
- File size limits
- Input sanitization

## 📝 Usage Example

### Analyzing Plant Health

```javascript
// Frontend API call
const formData = new FormData();
formData.append('image', imageFile);

const response = await axios.post('http://localhost:5002/predict', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
});

console.log(response.data.results);
// {
//   species: "Tomato___Late_blight",
//   confidence: 0.95,
//   yellow_percentage: 12.5,
//   spot_count: 25,
//   estimated_compost_grams: 8.75
// }
```

### Analyzing Grass Health

```javascript
const formData = new FormData();
formData.append('image', grassImage);

const response = await axios.post('http://localhost:5000/analyze_grass', formData);

console.log(response.data.results);
// {
//   healthy_percentage: 75.2,
//   unhealthy_percentage: 24.8,
//   compost_needed_grams: 248.0,
//   analysis_details: "Grass is moderately healthy..."
// }
```

## 🐛 Troubleshooting

### Common Issues

**Model not loading:**
- Ensure `plant_species_model.h5` and `class_indices.json` exist
- Check TensorFlow installation: `pip install tensorflow`
- Verify Python version compatibility

**CORS errors:**
- Check that all APIs are running
- Verify port numbers (5000, 5002, 5173)
- Review CORS middleware configuration

**Image upload failures:**
- Check file size limits
- Verify allowed file extensions (png, jpg, jpeg)
- Ensure upload directories exist and are writable

**Port already in use:**
```bash
# Windows: Find and kill process
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5000 | xargs kill -9
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras** for machine learning capabilities
- **OpenCV** for image processing
- **FastAPI** for the excellent Python web framework
- **shadcn/ui** for beautiful UI components
- **Together AI** for LLM integration
- Plant disease dataset contributors

## 📧 Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.

## 🔮 Future Enhancements

- [ ] Mobile app version
- [ ] Real-time analysis with camera feed
- [ ] Multi-language support
- [ ] Cloud deployment
- [ ] Advanced disease classification
- [ ] Treatment tracking and reminders
- [ ] Community features for sharing insights
- [ ] Integration with IoT sensors
- [ ] Weather data correlation
- [ ] Crop yield prediction

---

**Built with ❤️ for sustainable agriculture and plant health**

