from typing import Optional
from ..base_llm_evaluator import BaseLlmEvaluator


class FewShotExampleFaithfulness:
    """
    Class represting an example of the evaluation that could be used for few-shot prompting.
    """

    # User's question
    context: str
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
        context: str,
        response: str,
        eval_function: str,
        eval_result: str,
        eval_reason: str,
    ):
        """
        Initialize a new instance of FewShotExample.
        """
        self.context = context
        self.response = response
        self.eval_function = eval_function
        self.eval_result = eval_result
        self.eval_reason = eval_reason


class Faithfulness(BaseLlmEvaluator):
    """
    This class determines whether the chatbot's answer hether the response can be inferred using only the information provided as context.

    Attributes:
        open_ai_completion (OpenAICompletion): Instance for interactions with OpenAI's API.
        athina_api_key (str): API key for Athina.
        metadata (dict): Metadata for logging.
        examples (list[FewShotExampleFaithfulness]): List of few-shot examples used for evaluation.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
        You are an expert at evaluating whether the response can be inferred using the information provided as context.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        context: {}.
        response:{}.
        2. Make sure to also consider these instructions: {}
        3. Determine if the response can be inferred from the context provided.
        4. Provide a brief explanation of what information the response contained that was not provided to it in the context, labeled as 'explanation', leading up to a verdict (Yes/No) labeled as 'verdict'.
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

    # Pre-defined prompts for OpenAI's GPT model
    def system_message(self):
        return self.SYSTEM_MESSAGE_TEMPLATE

    def user_message(self, context, response):
        if self.additional_instructions is None:
            self.additional_instructions = ""

        return self.USER_MESSAGE_TEMPLATE.format(
            context, response, self.additional_instructions, self.examples
        )

    def evaluate(self, context: str, response: str):
        """
        Evaluation for is response faithful to context
        """
        user_message = self.user_message(context, response)
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

    @staticmethod
    def get_few_shot_examples():
        """
        Returns the few-shot examples.
        """
        # Creating instances of the FewShotExampleCcei class for each example
        example1 = FewShotExampleFaithfulness(
            context="Y Combinator is a startup accelerator launched in March 2005. It has been used to launch more than 4,000 companies",
            response="125,000",
            eval_function="is_response_faithful_to_context",
            eval_result="No",
            eval_reason="The context does not contain any information to substantiate the response.",
        )

        # Joining the string representations of the instances
        examples = "\n\n".join([str(example1)])
        return examples
