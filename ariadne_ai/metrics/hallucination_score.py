from .metric import Metric

class HallucinationScore(Metric):
    """
    Calculates the hallucination score between two sets of answers.

    HallucinationScore computes the proportion of summaries where a question generated 
    from the summary receives a 'Yes/No' answer from the summary, but an 'Unknown' answer 
    from the source document. A high score indicates potential content in the summary 
    that is absent from the source document.
    """

    @staticmethod
    def _compute_metric(answers_src, answers_sum):
        """
        Computes the number of hallucinations between the answers from source and summary.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.

        Returns:
            int: Number of questions indicating hallucinations.
        """
        answers_src_ls = list(answers_src.values())
        answers_sum_ls = list(answers_sum.values())
        n_hallucination = 0
        for ans_src, ans_sum in zip(answers_src_ls, answers_sum_ls):
            if ans_src.strip().lower() == 'unknown' and ans_sum.strip().lower() in ['yes', 'no']:
                n_hallucination += 1
        return n_hallucination

    @staticmethod
    def compute(answers_src, answers_sum, n_questions):
        """
        Computes the hallucination score.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.
            n_questions (int): Total number of questions.

        Returns:
            float: Hallucination score.
        """
        n_hallucination = HallucinationScore._compute_metric(answers_src, answers_sum)
        hal_score = n_hallucination / n_questions
        return hal_score
