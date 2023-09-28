from .metric import Metric

class Faithfullness(Metric):
    """
    The Faithfullness class is a metric that determines if the response can be inferred purely from the context provided.
    """

    @staticmethod
    def verdict_to_int(verdict: str) -> int:
        """
        Converts the verdict to an integer score. 'yes' verdict is converted to 1, 'no' verdict is converted to 0, and any other verdict is converted to None.
        """
        verdict = verdict.lower()
        score = 1 if verdict == 'yes' else 0 if verdict == 'no' else None
        return score
    
    @staticmethod
    def compute(faith_eval):
        """
        Computes the faithfullness metric.
        """
        is_faithfull = Faithfullness.verdict_to_int(faith_eval['verdict'])
        explanation = faith_eval['explanation']
        return is_faithfull, explanation