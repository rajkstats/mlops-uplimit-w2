import json
import requests


from collections import Counter


# Test functionality
def test_prediction():
    ##Testing Prediction

    data = {"review": "Hello world this is the best product ever!"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(data),
        headers=headers
    )
    print("Testing Review:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")

# Canary test functionality
def test_canary():
    ##Test canary deployment distribution"""

    data = {"review": "Hello world this is the best product ever!"}
    headers = {"Content-Type": "application/json"}

    version_counts = Counter()

    # Number of requests to test canary
    num_requests = 50

    print("Canary Deployment Test:")
    for i in range(num_requests):
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            data=json.dumps(data),
            headers=headers
        )
        result = response.json()
        version_counts[result.get('model_version', 'unknown')] += 1

    print("Traffic Distribution:")
    for version, count in version_counts.items():
        print(f"{version}: {(count/num_requests)*100:.1f}%")

if __name__ == "__main__":

    # API Test functionality
    test_prediction()

    # Run canary test
    test_canary()