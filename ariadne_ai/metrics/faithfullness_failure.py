from .metric import Metric

class FaithfullnessFailure(Metric):
    """
    The Faithfullness class is a metric that determines if the response can be inferred purely from the context provided.
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
    def compute(faith_eval):
        """
        Computes the faithfullness metric.
        """
        is_faithfullness_failure = FaithfullnessFailure.verdict_to_int(faith_eval['verdict'])
        explanation = faith_eval['explanation']
        return is_faithfullness_failure, explanation