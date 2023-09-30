from ..metric import Metric

class ContextRelevanceFailure(Metric):
    """
    The ContextRelevanceFailure class is a metric that determines if  a chatbot can answer a user's query using ONLY the information provided to you as context.
    """

    @staticmethod
    def verdict_to_int(verdict: str) -> int:
        """
        Converts the verdict to an integer score. 'yes' verdict is considered non-failure, while 'no' verdict is considered failure.
        """
        verdict = verdict.lower()
        score = 1 if verdict == 'no' else 0 if verdict == 'yes' else None
        return score
    
    @staticmethod
    def compute(context_relevance_eval):
        """
        Computes the context relevance metric.
        """
        is_context_relevance_failure = ContextRelevanceFailure.verdict_to_int(context_relevance_eval['verdict'])
        explanation = context_relevance_eval['explanation']
        return is_context_relevance_failure, explanation