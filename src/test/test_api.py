from unittest import TestCase
from api import app

class TestAPI(TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_endpoint(self):
        response = self.client.get('/predict?number=2')
        self.assertIsNotNone(response)
