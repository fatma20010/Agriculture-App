"""
Example script demonstrating how to use the authentication API.
This shows the flow for:
1. Creating a new user account
2. Logging in to get a JWT token
3. Using that token to access protected resources
4. Viewing user history

To use in your React frontend, you would implement similar API calls using axios or fetch.
"""

import requests
import json
import os

# API base URL
BASE_URL = "http://localhost:5002"

def register_user(username, email, password, full_name=None):
    """Register a new user"""
    url = f"{BASE_URL}/register"
    data = {
        "username": username,
        "email": email,
        "password": password,
        "full_name": full_name,
        "is_admin": False
    }
    
    response = requests.post(url, json=data)
    print(f"Register response: {response.status_code}")
    if response.status_code == 201:
        print(f"User registered: {response.json()}")
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def login_user(username, password):
    """Login and get access token"""
    url = f"{BASE_URL}/token"
    data = {
        "username": username,
        "password": password,
        "scope": "user"  # Can be "user admin" for admin access
    }
    
    # Note: The OAuth2 spec requires this to be form data, not JSON
    response = requests.post(url, data=data)
    print(f"Login response: {response.status_code}")
    if response.status_code == 200:
        token_data = response.json()
        print(f"Login successful! Token type: {token_data['token_type']}")
        return token_data
    else:
        print(f"Login failed: {response.text}")
        return None

def get_user_profile(token):
    """Get user profile using token"""
    url = f"{BASE_URL}/users/me"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(url, headers=headers)
    print(f"User profile response: {response.status_code}")
    if response.status_code == 200:
        user_data = response.json()
        print(f"User profile: {json.dumps(user_data, indent=2)}")
        return user_data
    else:
        print(f"Error getting profile: {response.text}")
        return None

def get_user_history(token):
    """Get user analysis history"""
    url = f"{BASE_URL}/user/history"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(url, headers=headers)
    print(f"User history response: {response.status_code}")
    if response.status_code == 200:
        history = response.json()
        print(f"Found {len(history)} analysis records")
        for item in history:
            print(f"Analysis {item['id']} ({item['analysis_type']}) on {item['created_at']}")
            print(f"  Species: {item['result_summary']['species']}")
            print(f"  Confidence: {item['result_summary']['confidence']:.2f}")
            print(f"  Yellow %: {item['result_summary']['yellow_percentage']:.2f}")
            print()
        return history
    else:
        print(f"Error getting history: {response.text}")
        return None

def upload_image_with_auth(token, image_path):
    """Upload image for analysis with authentication"""
    url = f"{BASE_URL}/predict"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, headers=headers, files=files)
    
    print(f"Upload response: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis result: {json.dumps(result, indent=2)}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    print("\n1. REGISTER A NEW USER")
    print("=====================")
    register_user("testuser", "test@example.com", "password123", "Test User")
    
    print("\n2. LOGIN")
    print("========")
    token_data = login_user("testuser", "password123")
    
    if token_data:
        access_token = token_data["access_token"]
        
        print("\n3. GET USER PROFILE")
        print("=================")
        get_user_profile(access_token)
        
        print("\n4. GET USER HISTORY")
        print("=================")
        get_user_history(access_token)
        
        print("\n5. UPLOAD IMAGE")
        print("=============")
        # Change this to an actual image path
        test_image_path = "test.jpg"  # Use an existing image in your project
        if os.path.exists(test_image_path):
            upload_image_with_auth(access_token, test_image_path)
        else:
            print(f"Test image not found at {test_image_path}")
    else:
        print("Login failed, cannot continue with authenticated requests") 