from locust import HttpUser, task, between

class StressTest(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def test_image_to_text_endpoint(self):
        url = "/image_to_text"

        # Your text data
        text = "a cat, a dog, a giraffe"

        # Path to the image file you want to upload
        image_path = "test.jpg"

        # Create a dictionary for the files to be uploaded
        files = {
            "text": (None, text),
            "file": ("test.jpg", open(image_path, "rb"), "image/jpeg")
        }

        res = self.client.post(
            url=url,
            files=files
        )