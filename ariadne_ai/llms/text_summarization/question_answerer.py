from ..open_ai_completion import OpenAICompletion


class QuestionAnswerer:
    """
    This class determines whether the chatbot's answer was correct based on
    the given content and user's question.

    Attributes:
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
    """

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at responding to closed-ended (Yes/No) questions using ONLY the provided context.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
           Questions: {}.
           Context: {}.
        2. Respond to each question from the provided 'questions', using either 
           'Yes', 'No', or 'Unknown', based on the given context.
        3. Return a JSON object in the following format: "question1": "answer1", "question2": "answer2",...
    """

    def __init__(self, model, open_ai_key):
        """
        Initialize the QuestionAnswerer class.
        """
        self.openAIcompletion = OpenAICompletion(model, open_ai_key)

    def answer(self, questions: str, context: str) -> dict:
        """
        Respond to each question from the provided 'questions' given the context.

        Args:
            questions (str): A set of questions posed to the chatbot.
            context (str): Context used to inform the chatbot's answers.

        Returns:
            dict: Evaluation results formatted as a dictionary with questions as keys and
                  'Yes', 'No', or 'Unknown' as values.
        """

        user_message = self.USER_MESSAGE_TEMPLATE.format(questions, context)
        message = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(
            openai_response
        )

        return openai_response_json
