import time
import requests
import logging

# Configure logging to write to a log file
logging.basicConfig(filename='gpt/gpt_response_log.txt', level=logging.INFO)
log = logging.getLogger(__name__)

def make_api_call(input_text):
    url = "http://localhost:8080/infer"
    params = {'input_txt': input_text}
    max_retries = 3
    retry_delay = 10  # 0.5 seconds

    for i in range(max_retries):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for non-2xx responses
            return response
        except requests.exceptions.RequestException:
            time.sleep(retry_delay)
            continue
    raise ConnectionError("API call failed after multiple retries.")

def test_api_calls():
    total_time = 0
    num_calls = 100
    input_text = "Obliviate"

    for i in range(num_calls):
        start_time = time.time()
        response = make_api_call(input_text)
        end_time = time.time()

        response_time = end_time - start_time
        total_time += response_time

        logging.info(f"API Call {i + 1}: Response Time: {response_time:.3f} seconds")

    average_response_time = total_time / num_calls
    logging.info(f"Average Response Time after {num_calls} API calls: {average_response_time:.3f} seconds")

def main():
    test_api_calls()

if __name__ == "__main__":
    main()