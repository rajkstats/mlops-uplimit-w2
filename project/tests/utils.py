import requests

def get_label(text, predictor_url="http://127.0.0.1:8000/predict"):
    try:
        # Create the payload to send to the predictor
        payload = {"review": text}

        # Send POST request to the predictor API
        response = requests.post(predictor_url, json=payload)

        # Check if the request was successful
        response.raise_for_status()

        # Get the prediction from the API response
        result = response.json()

        # Log the entire result for debugging
        print(f"API response: {result}")

        # Check if the 'prediction' key is present
        if "label" in result:
            return result["label"].lower()
        else:
            raise ValueError(f"Prediction result is not in the expected format: {result}")

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")
