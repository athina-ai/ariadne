from ..open_ai_completion import OpenAICompletion


class FewShotExampleContextRelevance:
    """
    Class represting an example of the evaluation that could be used for few-shot prompting.
    """

    # Retrieved context
    context: str
    # User's question
    query: str
    # Name of the evaluation function
    eval_function: str
    # Evaluation result
    eval_result: str
    # LLM's reason for evaluation
    eval_reason: str

    def __init__(
        self,
        context: str,
        query: str,
        eval_function: str,
        eval_result: str,
        eval_reason: str,
    ):
        """
        Initialize a new instance of FewShotExample.
        """
        self.context = context
        self.query = query
        self.eval_function = eval_function
        self.eval_result = eval_result
        self.eval_reason = eval_reason


class ContextRelevance:
    """
    This class determines whether the chatbot's response can be inferred using only the information provided as context.

    Attributes:
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
        examples (list[FewShotExampleFaithfulness]): List of few-shot examples used for evaluation.
    """

    SYSTEM_MESSAGE = """ 
        You are an expert at evaluating whether a chatbot can answer a user's query using ONLY the information provided to you as context.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        user's query: {}.
        context:{}.
        2. Determine if the chatbot can answer the user's query with nothing but the "context" information provided to you.
        3. Provide a brief explanation of why the context does or does not contain sufficient information, labeled as 'explanation', leading up to a verdict (Yes/No) labeled as 'verdict'.
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

    def evaluate(self, query: str, context: str):
        """
        Evaluation for is response faithful to context
        """
        user_message = self.USER_MESSAGE_TEMPLATE.format(query, context, self.examples)
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
        # Creating instances of the FewShotExampleContextRelevance class for each example
        example1 = FewShotExampleContextRelevance(
            context="bjarne stroustrup invented C++",
            query="Who invented the linux os",
            eval_function="does_context_contain_sufficient_information",
            eval_result="No",
            eval_reason="The context does not provide any relevant information about the Linux OS or its inventor.",
        )
        example2 = FewShotExampleContextRelevance(
            context="In 1969, Neil Armstrong became the first person to walk on the moon.",
            query="What was the name of the spaceship used for the moon landing in 1969?",
            eval_function="does_context_contain_sufficient_information",
            eval_result="No",
            eval_reason="The context provided does not include any information about the name of the spaceship used for the moon landing. The query specifically asks for the name of the spaceship, which is not present in the context.",
        )
        # Joining the string representations of the instances
        examples = "\n\n".join([str(example1), str(example2)])
        return examples
