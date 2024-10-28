from io import BytesIO
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
from model import preprocess_img, predict_result

# Basic testing

# Test preprocess_img with a Simple RGB Image
def test_preprocess_img_standard_image():
    # Creates a 224x224 white RGB image in memory
    img = Image.new("RGB", (224, 224), color="white")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Run preprocess_img function
    processed_img = preprocess_img(img_bytes)

    # Evaluate processed image
    assert processed_img.shape == (1, 224, 224, 3), "Processed image has incorrect shape."
    assert (processed_img >= 0).all() and (processed_img <= 1).all(), "Processed image has incorrect pixel values."

# Test predict_result with All-Zero Predictions
@patch("model.model")
def test_predict_result_all_zero_predictions(mock_model):
    # Mock predict method to return all-zero probabilities
    mock_predict_output = np.array([[0.0, 0.0, 0.0]])
    mock_model.predict = MagicMock(return_value=mock_predict_output)

    # Run predict_result with a dummy input
    dummy_input = np.zeros((1, 224, 224, 3))
    result = predict_result(dummy_input)

    # Evaluate result
    assert result == 0, "predict_result did not return index 0 for all-zero probabilities."


# Advanced Testing

# Test preprocess_img with a Large Image
def test_preprocess_img_large_image():
    # Creates a large 1000x1000 RGB image in memory
    img = Image.new("RGB", (1000, 1000), color="blue")
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    # Run preprocess_img function
    processed_img = preprocess_img(img_bytes)

    # Evaluate processed image
    assert processed_img.shape == (1, 224, 224, 3), "Processed large image has incorrect shape."
    assert (processed_img >= 0).all() and (processed_img <= 1).all(), "Processed large image has incorrect pixel values."

# Test predict_result with High Probability Ties
@patch("model.model")
def test_predict_result_high_probability_ties(mock_model):
    # Mock predict method with tied probabilities for two classes
    mock_predict_output = np.array([[0.4, 0.7, 0.7, 0.2]])
    mock_model.predict = MagicMock(return_value=mock_predict_output)

    # Run predict_result with a dummy input
    dummy_input = np.zeros((1, 224, 224, 3))
    result = predict_result(dummy_input)

    # Evaluate result
    assert result == 1, "predict_result did not return the first index of the highest probability in case of ties."
