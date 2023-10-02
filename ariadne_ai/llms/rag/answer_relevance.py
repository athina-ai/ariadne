from ..open_ai_completion import OpenAICompletion


class FewShotExampleAnswertRelevance:
    """
    Class represting an example of the evaluation that could be used for few-shot prompting.
    """

    # User's question
    query: str
    # User's question
    response: str
    # Name of the evaluation function
    eval_function: str
    # Evaluation result
    eval_result: str
    # LLM's reason for evaluation
    eval_reason: str

    def __init__(
        self,
        query: str,
        response: str,
        eval_function: str,
        eval_result: str,
        eval_reason: str,
    ):
        """
        Initialize a new instance of FewShotExample.
        """
        self.query = query
        self.response = response
        self.eval_function = eval_function
        self.eval_result = eval_result
        self.eval_reason = eval_reason

    def __str__(self):
        """
        Return a string representation of the FewShotExample.
        """
        return (
            f"Query: {self.query}\n"
            f"Response: {self.response}\n"
            f"{self.eval_function}: {self.eval_result}\n"
            f"Reason:{self.eval_reason}"
        )


class AnswerRelevance:
    """
    This class determines whether the chatbot's response answers specifically what the user is asking about, and covers all aspects of the user's query

    Attributes:
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
        examples (list[FewShotExampleFaithfulness]): List of few-shot examples used for evaluation.
    """

    SYSTEM_MESSAGE = """ 
        You are an expert at evaluating whether a response answers a user's query sufficiently.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        user's query: {}.
        response:{}.
        2. Determine if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
        3. Provide a brief explanation of why the response does or does not answer the user's query sufficiently, labeled as 'explanation', leading up to a verdict (Yes/No) labeled as 'verdict'.
        4. Return a JSON object in the following format: "verdict": 'verdict', "explanation": 'explanation'.

        Here's are some examples: 
        {}
    """

    def __init__(self, model, open_ai_key):
        """
        Initialize the QuestionAnswerer class.
        """
        self.openAIcompletion = OpenAICompletion(model, open_ai_key)
        self.examples = self.get_few_shot_examples()

    def evaluate(self, query: str, response: str):
        """
        Evaluation for is response faithful to context
        """
        user_message = self.USER_MESSAGE_TEMPLATE.format(query, response, self.examples)
        system_message = self.SYSTEM_MESSAGE
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(
            openai_response
        )
        return openai_response_json

    # Few shot examples
    @staticmethod
    def get_few_shot_examples():
        """
        Returns the few-shot examples.
        """
        # Creating instances of the FewShotExampleCcei class for each example
        example1 = FewShotExampleAnswertRelevance(
            query="How much does Y Combinator invest in startups",
            response="125,000",
            eval_function="does_response_answer_query",
            eval_result="Yes",
            eval_reason="The response is a reasonable answer to the query",
        )
        example2 = FewShotExampleAnswertRelevance(
            query="What was the name of the spaceship to first land on the moon",
            response="Neil Armstrong was the first astronaut on the moon",
            eval_function="does_response_answer_query",
            eval_result="No",
            eval_reason="The response does not answer the query asking about the name of the spaceship.",
        )
        example3 = FewShotExampleAnswertRelevance(
            query="Will alicia keys be at the festival",
            response="Neil Armstrong was the first astronaut on the moon",
            eval_function="does_response_answer_query",
            eval_result="Yes",
            eval_reason="The response is a reasonable answer to the query.",
        )
        # Joining the string representations of the instances
        examples = "\n\n".join([str(example1), str(example2), str(example3)])
        return examples
