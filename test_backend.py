"""
Test Script for Medical Report Analyzer
========================================
This script tests your backend API to ensure it's working correctly.

Usage:
    python test_backend.py

Make sure your backend is running before testing!
"""

import requests
import json
from pathlib import Path

# Configuration - Change this to match your backend
BACKEND_URL = "http://localhost:5000"  # or http://localhost:8000 for FastAPI

def test_health_check():
    """Test if the backend is running"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/health")
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend!")
        print(f"   Make sure the server is running at {BACKEND_URL}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_file_upload():
    """Test file upload endpoint"""
    print("\n" + "="*60)
    print("TEST 2: File Upload (with dummy file)")
    print("="*60)
    
    # Create a dummy text file for testing
    dummy_content = """
    Medical Report
    ==============
    Patient: Test Patient
    Date: 2024-01-01
    
    Test results indicate normal levels.
    No abnormalities detected.
    """
    
    try:
        # Create temporary file
        files = {
            'report': ('test_report.txt', dummy_content, 'text/plain')
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/analyze/file",
            files=files
        )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response:")
            print(f"   - Detected: {result.get('detected')}")
            print(f"   - Confidence: {result.get('confidence')}%")
            print(f"   - Disease: {result.get('diseaseName')}")
            print(f"   - Type: {result.get('reportType')}")
            return True
        else:
            print(f"❌ Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_image_upload():
    """Test image upload endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Image Upload (with dummy image)")
    print("="*60)
    
    try:
        from PIL import Image
        import io
        
        # Create a dummy image for testing
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {
            'report': ('test_image.png', img_bytes, 'image/png')
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/analyze/image",
            files=files
        )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response:")
            print(f"   - Detected: {result.get('detected')}")
            print(f"   - Confidence: {result.get('confidence')}%")
            print(f"   - Disease: {result.get('diseaseName')}")
            print(f"   - Type: {result.get('reportType')}")
            return True
        else:
            print(f"❌ Error Response: {response.text}")
            return False
            
    except ImportError:
        print("⚠️  Skipping: PIL not installed (run: pip install pillow)")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_invalid_file():
    """Test error handling with invalid file"""
    print("\n" + "="*60)
    print("TEST 4: Invalid File Handling")
    print("="*60)
    
    try:
        # Try to upload a file with invalid extension
        files = {
            'report': ('test.xyz', b'invalid content', 'application/octet-stream')
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/analyze/file",
            files=files
        )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 400:
            print(f"✅ Correctly rejected invalid file")
            print(f"   Error message: {response.json().get('error')}")
            return True
        else:
            print(f"⚠️  Expected 400 status code, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("MEDICAL REPORT ANALYZER - BACKEND TESTS")
    print("="*60)
    print(f"Testing backend at: {BACKEND_URL}")
    
    results = {
        'Health Check': test_health_check(),
        'File Upload': test_file_upload(),
        'Image Upload': test_image_upload(),
        'Invalid File Handling': test_invalid_file()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is None:
            status = "⚠️  SKIPPED"
        else:
            status = "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed! Your backend is ready to use.")
        print("\nNext steps:")
        print("1. Open report-analyzer.html in your browser")
        print("2. Upload a real medical report file or image")
        print("3. Integrate your actual ML models")
    elif failed > 0:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*60)


if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import requests
    except ImportError:
        print("❌ ERROR: 'requests' library not installed")
        print("   Run: pip install requests")
        exit(1)
    
    run_all_tests()
