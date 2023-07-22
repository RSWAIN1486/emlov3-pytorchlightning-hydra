import unittest
import json

from gradio_client import Client

class TestGPTGradio(unittest.TestCase):
    @classmethod

    def test_predict(self):
        client = Client('http://localhost:80')
        result = client.predict(
            """THE RIDDLE HOUSE 

            The villagers of Little Hangleton still called it “the 
            Riddle House,” even though it""",
            512,  # Here, we pass the additional value output_tokens
            api_name="/predict"
        )
        print(f"got result = {result}")


if __name__ == '__main__':
    unittest.main()
