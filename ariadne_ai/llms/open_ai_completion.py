import openai
import time
import traceback
import json
from typing import Optional
from athina_logger.inference_logger import InferenceLogger
from athina_logger.api_key import AthinaApiKey
from athina_logger.exception.custom_exception import CustomException


class OpenAICompletion:
    """
    A class to interact with OpenAI's ChatCompletion API.

    Attributes:
    - model (str): The model to use for completions, default is "gpt-3.5-turbo".
    - open_ai_key (str): The API key for OpenAI.
    - temperature (float): OpenAI temperature setting.
    - max_tokens (int): OpenAI maximum number of tokens setting. The token count of your prompt plus max_tokens cannot exceed the model's context.
    """

    def __init__(
        self,
        model: str,
        open_ai_key: str,
        athina_api_key: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Initializes the OpenAICompletion with the provided settings.
        """
        # Setting instance attributes based on provided parameters or defaults
        self.model = model
        self.metadata = metadata
        self.open_ai_key = open_ai_key
        AthinaApiKey.set_api_key(athina_api_key)

        # Setting the API key for OpenAI based on provided key
        openai.api_key = self.open_ai_key

    def get_completion_from_messages(
        self,
        messages,
        temperature: float = 0,
        max_tokens: int = 2000,
        retry_count: int = 0,
    ):
        """
        Fetches a completion response from OpenAI's ChatCompletion API based on the provided messages.
        """
        try:
            # Attempting to fetch a response from OpenAI
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # Logging the response to Athina
            if self.metadata is None:
                environment = None
                prompt_slug = None
                customer_id = None
                customer_user_id = None
                external_reference_id = None
                session_id = None
            else:
                environment = (
                    self.metadata["environment"]
                    if self.metadata["environment"] is not None
                    else "default"
                )
                prompt_slug = (
                    self.metadata["prompt_slug"]
                    if self.metadata["prompt_slug"] is not None
                    else "default"
                )
                customer_id = (self.metadata["customer_id"],)
                customer_user_id = (self.metadata["customer_user_id"],)
                external_reference_id = (self.metadata["external_reference_id"],)
                session_id = (self.metadata["session_id"],)

            if AthinaApiKey.get_api_key() is not None:
                try:
                    InferenceLogger.log_open_ai_chat_response(
                        prompt_slug=prompt_slug,
                        messages=messages,
                        model=self.model,
                        completion=response,
                        context=None,
                        response_time=response_time_ms,
                        customer_id=customer_id,
                        customer_user_id=customer_user_id,
                        external_reference_id=external_reference_id,
                        session_id=session_id,
                        environment=environment,
                    )
                except Exception as e:
                    print("Failed to log to Athina", e)

        except openai.error.RateLimitError as e:
            print("RateLimitError", e)
            # Calculate the wait time using exponential backoff
            base_wait_time = 15
            max_retries = 3
            wait_time = base_wait_time * (2**retry_count)

            # Wait for the calculated wait time
            time.sleep(wait_time)

            if retry_count < max_retries:
                return self.get_completion_from_messages(
                    messages, temperature, max_tokens, retry_count + 1
                )
            else:
                print("Max retries reached - unable to complete OpenAI request")
                raise e
        except openai.error.AuthenticationError as e:
            raise openai.error.AuthenticationError("Please pass a valid OpenAi key.")
        except openai.error.Timeout as e:
            print("Timeout", e)
            # In case of a rate limit error, wait for 15 seconds and retry
            time.sleep(15)
            if retry_count < 3:
                return self.get_completion_from_messages(
                    messages, retry_count=retry_count + 1
                )
            else:
                print("Max retries reached - unable to complete OpenAI request")
                raise e
        except openai.error.InvalidRequestError as e:
            print("InvalidRequestError", e)
            raise e
        except openai.error.APIConnectionError as e:
            # In case of a api connection error, wait for 60 seconds and retry
            time.sleep(30)
            print("APIConnectionError", e)
            if retry_count < 3:
                return self.get_completion_from_messages(
                    messages, retry_count=retry_count + 1
                )
            else:
                print("Max retries reached - unable to complete OpenAI request")
                raise e
        except Exception as e:
            print("Exception", e)
            traceback.print_exc()
            return None
        return response.choices[0].message["content"]

    @staticmethod
    def _extract_json(data_string: str) -> str:
        """
        Extracts a JSON string from a larger string.
        Assumes the JSON content starts with '{' and continues to the end of the input string.
        """
        try:
            start_index = data_string.index("{")
            end_index = data_string.rfind("}")
            json_string = data_string[start_index : end_index + 1]
        except Exception as e:
            print("Failed to extract json", e)
            json_string = data_string
        return json_string

    @staticmethod
    def _load_json_from_text(text):
        """
        Extracts and loads a JSON string from a given text.
        """
        try:
            data = json.loads(text)
        except json.decoder.JSONDecodeError as e:
            print("Failed to load JSON from text", e)
            data = None
        return data

    @staticmethod
    def extract_json_from_response(response):
        # In case you cannot handle an error, return None
        if response is None:
            return None
        response_json_format = OpenAICompletion._extract_json(response)
        response_json = OpenAICompletion._load_json_from_text(response_json_format)
        return response_json
