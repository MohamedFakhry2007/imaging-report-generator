import requests
import pytest
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the backend URL from environment
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL')
if not BACKEND_URL:
    raise ValueError("REACT_APP_BACKEND_URL environment variable not set")

class TestImageToStoryAPI:
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = requests.get(f"{BACKEND_URL}/api")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        print("✅ Root endpoint test passed")

    def test_generate_story_with_valid_image(self):
        """Test story generation with a valid image"""
        # Create a test image path
        test_image_path = Path(__file__).parent / "test_image.jpg"
        
        if not test_image_path.exists():
            print("⚠️ Test image not found, skipping story generation test")
            return

        with open(test_image_path, "rb") as image_file:
            files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
            response = requests.post(f"{BACKEND_URL}/api/generate-story", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert "story" in data
            assert isinstance(data["story"], str)
            assert len(data["story"]) > 0
            print("✅ Story generation test passed")

    def test_generate_story_with_invalid_file(self):
        """Test story generation with an invalid file"""
        # Create an invalid file (text instead of image)
        files = {"file": ("test.txt", b"This is not an image", "text/plain")}
        response = requests.post(f"{BACKEND_URL}/api/generate-story", files=files)
        
        assert response.status_code == 400
        print("✅ Invalid file handling test passed")

if __name__ == "__main__":
    print("🔍 Starting API tests...")
    
    test_instance = TestImageToStoryAPI()
    
    try:
        test_instance.test_root_endpoint()
        test_instance.test_generate_story_with_valid_image()
        test_instance.test_generate_story_with_invalid_file()
        print("\n✨ All API tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
