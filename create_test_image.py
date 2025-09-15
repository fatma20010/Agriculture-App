import numpy as np
import cv2

# Create a simple test image
def create_test_image():
    # Create a blank image with green background
    img = np.ones((224, 224, 3), dtype=np.uint8) * np.array([0, 200, 0], dtype=np.uint8)
    
    # Add some yellow spots to simulate plant disease
    for i in range(10):
        x = np.random.randint(20, 204)
        y = np.random.randint(20, 204)
        radius = np.random.randint(5, 15)
        cv2.circle(img, (x, y), radius, (0, 255, 255), -1)  # Yellow spots
    
    # Add some brown spots
    for i in range(5):
        x = np.random.randint(20, 204)
        y = np.random.randint(20, 204)
        radius = np.random.randint(3, 10)
        cv2.circle(img, (x, y), radius, (0, 100, 150), -1)  # Brown spots
    
    # Save the image
    cv2.imwrite("test.jpg", img)
    print("Test image created: test.jpg")

if __name__ == "__main__":
    create_test_image() 