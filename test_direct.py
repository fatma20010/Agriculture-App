import requests
import os
import sys
import time

def test_server_connection():
    """Test if the server is running and responding correctly."""
    print("Testing server connection...")
    try:
        response = requests.get("http://127.0.0.1:5001/", timeout=5)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
        print(f"Headers: {response.headers}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Connection error: Server is not running or not accessible")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_image_upload():
    """Test uploading an image to the server."""
    print("\nTesting image upload...")
    
    # Find a test image
    test_images = [
        "test.jpg", "test.png", "sample.jpg", "sample.png", 
        "image.jpg", "image.png", "plant.jpg", "plant.png"
    ]
    
    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if not test_image:
        print("No test image found. Please place a test image in the current directory.")
        return False
    
    print(f"Using test image: {test_image}")
    
    try:
        with open(test_image, "rb") as f:
            files = {"image": (test_image, f, "image/jpeg")}
            
            # Add delay to ensure server is ready
            time.sleep(1)
            
            # Set explicit timeout
            response = requests.post(
                "http://127.0.0.1:5001/predict",
                files=files,
                timeout=10
            )
        
        print(f"Status code: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Response: {response.json() if response.status_code == 200 else response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error uploading image: {e}")
        return False

if __name__ == "__main__":
    print("=== Server Test Tool ===")
    server_ok = test_server_connection()
    
    if server_ok:
        print("\n✅ Server is running")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--upload":
            upload_ok = test_image_upload()
            if upload_ok:
                print("\n✅ Image upload successful")
            else:
                print("\n❌ Image upload failed")
    else:
        print("\n❌ Server is not running or not responding correctly")
        print("\nTry running: python -m uvicorn veg:app --host 0.0.0.0 --port 5001") 