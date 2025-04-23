import unittest
import requests
import os
from PIL import Image
from io import BytesIO

class TestImageToArabicStoryAPI(unittest.TestCase):
    def setUp(self):
        # Get the backend URL from environment variable
        self.base_url = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')
        
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_root_endpoint(self):
        """Test the root endpoint returns correct message"""
        response = requests.get(f"{self.base_url}/api")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())

    def test_generate_story_valid_image(self):
        """Test story generation with valid image"""
        files = {
            'file': ('test.jpg', self.image_bytes, 'image/jpeg')
        }
        response = requests.post(f"{self.base_url}/api/generate-story", files=files)
        
        print("Story Generation Response:", response.json() if response.ok else response.text)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("story", response.json())
        self.assertTrue(isinstance(response.json()["story"], str))
        self.assertTrue(len(response.json()["story"]) > 0)

    def test_generate_story_no_image(self):
        """Test story generation without image"""
        response = requests.post(f"{self.base_url}/api/generate-story")
        self.assertEqual(response.status_code, 422)  # FastAPI validation error

    def test_generate_story_invalid_image(self):
        """Test story generation with invalid image data"""
        files = {
            'file': ('test.txt', b'not an image', 'text/plain')
        }
        response = requests.post(f"{self.base_url}/api/generate-story", files=files)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()