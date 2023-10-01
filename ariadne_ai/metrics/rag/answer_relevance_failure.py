from ..metric import Metric

class AnswerRelevanceFailure(Metric):
    """
    The AnswerRelevanceFailure class is a metric that determines whether a response answers a user's query sufficiently.
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
    def compute(answer_relevance_eval):
        """
        Computes the context relevance metric.
        """
        is_answer_relevance_failure = AnswerRelevanceFailure.verdict_to_int(answer_relevance_eval['verdict'])
        explanation = answer_relevance_eval['explanation']
        return is_answer_relevance_failure, explanation