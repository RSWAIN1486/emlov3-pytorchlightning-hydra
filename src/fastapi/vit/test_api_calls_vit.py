import requests
import logging
import time

# Configure logging to write to a log file
logging.basicConfig(filename='vit/vit_response_log.txt', level=logging.INFO)
log = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"
image_file = 'vit/img001.png'

def get_image_data(image_file):
    # Replace with the path to a sample image file for testing
    with open(image_file, "rb") as f:
        return f.read()

def make_api_call(image_data):
    url = f"{BASE_URL}/infer"
    
    files = {"image": (image_file, image_data, "image/png")}
    max_retries = 3
    retry_delay = 10  # 0.5 seconds

    for i in range(max_retries):
        try:
            start_time = time.time()
            response = requests.get(url, files=files)
            end_time = time.time()
            response_time = end_time - start_time
            response.raise_for_status()  # Raise an exception for non-2xx responses
            return response, response_time
        except requests.exceptions.RequestException:
            time.sleep(retry_delay)
            continue
    raise ConnectionError("API call failed after multiple retries.")


def test_api_calls():
    num_calls = 100
    total_response_time = 0

    # load classes
    with open('vit/cifar10_classes.txt', "r") as f:
            categories = [s.strip() for s in f.readlines()]

    for i in range(num_calls):
        response, response_time = make_api_call(get_image_data(image_file))
        total_response_time += response_time

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert isinstance(data, dict)

        # Add more assertions as needed based on the expected response format
        assert all(class_name in data for class_name in categories)

        # Log the response time for each API call
        logging.info(f"API Call {i + 1}: Response Time: {response_time:.3f} seconds")

    average_response_time = total_response_time / num_calls
    logging.info(f"Average Response Time after {num_calls} API calls: {average_response_time:.3f} seconds")

def test_health_api():
    url = f"{BASE_URL}/health"
    response = requests.get(url)

    assert response.status_code == 200
    assert response.json() == {"message": "ok"}


def main():
    test_api_calls()

if __name__ == "__main__":
    main()