from typing import Optional
from ..base_llm_evaluator import BaseLlmEvaluator


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


class AnswerRelevance(BaseLlmEvaluator):
    """
    This class determines whether the chatbot's response answers specifically what the user is asking about, and covers all aspects of the user's query

    Attributes:
        open_ai_completion (OpenAICompletion): Instance for interactions with OpenAI's API.
        athina_api_key (str): API key for Athina.
        metadata (dict): Metadata for logging.
        examples (list[FewShotExampleFaithfulness]): List of few-shot examples used for evaluation.
    """

    SYSTEM_MESSAGE_TEMPLATE = """ 
        You are an expert at evaluating whether a response answers a user's query sufficiently.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        user's query: {}.
        response:{}.
        2. Make sure to also consider these instructions: {}
        3. Determine if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
        4. Provide a brief explanation of why the response does or does not answer the user's query sufficiently, labeled as 'explanation', leading up to a verdict (Yes/No) labeled as 'verdict'.
        5. Return a JSON object in the following format: "verdict": 'verdict', "explanation": 'explanation'.

        Here's are some examples: 
        {}
    """

    def __init__(
        self,
        model,
        open_ai_key,
        athina_api_key: Optional[str] = None,
        metadata: Optional[dict] = None,
        additional_instructions: Optional[str] = None,
    ):
        super().__init__(
            model,
            open_ai_key=open_ai_key,
            athina_api_key=athina_api_key,
            metadata=metadata,
        )
        self.examples = self.get_few_shot_examples()
        self.additional_instructions = additional_instructions

    def system_message(self):
        return self.SYSTEM_MESSAGE_TEMPLATE

    def user_message(self, context, response):
        if self.additional_instructions is None:
            self.additional_instructions = ""
        return self.USER_MESSAGE_TEMPLATE.format(
            context, response, self.additional_instructions, self.examples
        )

    def evaluate(self, query: str, response: str):
        """
        Evaluation for is response faithful to context
        """
        user_message = self.user_message(query, response)
        system_message = self.system_message()
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        openai_response = self.open_ai_completion.get_completion_from_messages(message)
        openai_response_json = self.open_ai_completion.extract_json_from_response(
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
        # Joining the string representations of the instances
        examples = "\n\n".join([str(example1), str(example2)])
        return examples
