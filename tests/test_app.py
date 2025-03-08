import os
import sys
import pytest
from unittest.mock import patch

# Add parent directory to path to allow importing app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Flask app
try:
    from app import app
except ImportError:
    # If app can't be imported, we'll skip the tests
    app = None

# Use this fixture to get a test client for the app
@pytest.fixture
def client():
    if app is None:
        pytest.skip("Could not import app, skipping tests")
    
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_app_initialized():
    """Test if the app can be imported and initialized"""
    assert app is not None, "Flask app could not be imported"

def test_health_endpoint(client):
    """Test the health endpoint responds with 200 status"""
    response = client.get('/health')
    assert response.status_code == 200
    assert b'ok' in response.data.lower()

@patch('app.send_guardian_request')  # Mock API call to avoid rate limiting
def test_guardian_endpoint_returns_json(mock_send_guardian_request, client):
    """Test that the Guardian API endpoint returns JSON"""
    # Configure the mock to return a sample response
    mock_send_guardian_request.return_value = {
        'response': {
            'status': 'ok',
            'results': [
                {
                    'id': 'test-id',
                    'webTitle': 'Test Article',
                    'webPublicationDate': '2023-05-01T12:00:00Z',
                    'webUrl': 'https://example.com/article'
                }
            ]
        }
    }
    
    response = client.get('/api/guardian/news')
    assert response.status_code == 200
    assert response.content_type == 'application/json'
    data = response.get_json()
    assert 'articles' in data 