
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image
from model import preprocess_img, predict_result

# Test preprocess_img
def test_preprocess_img():
    # Creates a white image in memory
    img = Image.new("RGB", (500, 500), color="white")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Run the function using the in memory image
    processed_img = preprocess_img(img_bytes)

    # Evaluate shape of image
    assert processed_img.shape == (1, 224, 224, 3), "Processed image has incorrect shape."
    # Evaluates that image values are within range
    assert (processed_img >= 0).all() and (processed_img <= 1).all(), "Processed image has incorrect pixel values."

# Test predict_result
@patch("model.model")  # Mock the model
def test_predict_result(mock_model):
    # Mock predict method
    mock_predict_output = np.array([[0.1, 0.2, 0.7]])  # Example output
    mock_model.predict = MagicMock(return_value=mock_predict_output)

    # Run predict_result with a dummy input
    dummy_input = np.zeros((1, 224, 224, 3))  # Shape matches expected input
    result = predict_result(dummy_input)

    # Evaluate the prediction output
    assert result == 2, "predict_result returned an unexpected class index."
