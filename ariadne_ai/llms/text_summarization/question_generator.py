from ..open_ai_completion import OpenAICompletion

class QuestionGenerator:
    """
    Generates closed-ended (Yes/No) questions given a  text.
    
    Attributes:
        n_questions (int): Number of questions to generate.
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
    """

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at generating closed-ended (Yes/No) questions given the content of a text.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the text: {}.
        2. Generate {} closed-ended (Yes/No) questions based on the content.
        3. Return a JSON object in the following format: "question 1": 'Your question', "question 2": 'Your next question', ...
    """

    def __init__(self, model: str, n_questions: int, open_ai_key:str):
        """
        Initialize the QuestionGenerator.
        """
        self.n_questions = n_questions
        self.openAIcompletion = OpenAICompletion(model, open_ai_key)

    def generate(self, text: str) -> dict:
        """
        Generate a set of closed-ended questions based on the provided text.

        Args:
            text (str): The reference content used to generate questions.

        Returns:
            dict: A dictionary of generated questions with keys indicating the question order and values being the questions themselves.
        """
        user_message = self.USER_MESSAGE_TEMPLATE.format(text, self.n_questions)
        message = [
            {'role': 'system', 'content': self.SYSTEM_MESSAGE}, 
            {'role': 'user', 'content': user_message}
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(openai_response)

        return openai_response_json
