import openai
import time
from dotenv import load_dotenv

import os
import json

class OpenAICompletion:
    """
    A class to interact with OpenAI's ChatCompletion API.
    
    Attributes:
    - model (str): The model to use for completions, default is "gpt-3.5-turbo".
    - open_ai_key (str): The API key for OpenAI.
    - temperature (float): OpenAI temperature setting.
    - max_tokens (int): OpenAI maximum number of tokens setting. The token count of your prompt plus max_tokens cannot exceed the model's context.
    """
    def __init__(self, model, open_ai_key):
        """
        Initializes the OpenAICompletion with the provided settings.
        """
        # Setting instance attributes based on provided parameters or defaults
        load_dotenv()
        self.model = model
        if open_ai_key is None:
            self.open_ai_key = os.getenv("OPENAI_API_KEY")
        else:
            self.open_ai_key = open_ai_key
        
        # Setting the API key for OpenAI based on provided key
        openai.api_key = self.open_ai_key

    def get_completion_from_messages(self, messages, temperature=0, max_tokens = 2000):
        """
        Fetches a completion response from OpenAI's ChatCompletion API based on the provided messages.
        """
        try:
            # Attempting to fetch a response from OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.error.RateLimitError:
            # In case of a rate limit error, wait for 15 seconds and retry
            time.sleep(15)
            print('RateLimitError')
            return self.get_completion_from_messages(messages)
        except openai.error.AuthenticationError:
            raise openai.error.AuthenticationError("Please pass a valid OpenAi key.")
        except openai.error.Timeout:
            # In case of a rate limit error, wait for 15 seconds and retry
            time.sleep(15)
            print('Timeout')
            return self.get_completion_from_messages(messages)
        except openai.error.InvalidRequestError:
            print('InvalidRequestError')
            return None
        except openai.error.APIConnectionError:
            # In case of a api connection error, wait for 60 seconds and retry
            time.sleep(60)
            print('APIConnectionError')
            return self.get_completion_from_messages(messages)
        return response.choices[0].message["content"]

    @staticmethod
    def _extract_json(data_string: str) -> str:
        """
        Extracts a JSON string from a larger string. 
        Assumes the JSON content starts with '{' and continues to the end of the input string.
        """
        try:
            start_index = data_string.index("{")
            json_string = data_string[start_index:]
        except:
            json_string = data_string
        return json_string
    
    @staticmethod
    def _load_json_from_text(text):
        """
        Extracts and loads a JSON string from a given text.
        """
        try:
            data = json.loads(text)
        except json.decoder.JSONDecodeError:
            data = None
        return data
    
    @staticmethod
    def extract_json_from_response(response):
        # In case you cannot handle an error, return None
        if response is None:
            return None
        response_json_format = OpenAICompletion._extract_json(response)
        response_json = OpenAICompletion._load_json_from_text(response_json_format)
        return(response_json)