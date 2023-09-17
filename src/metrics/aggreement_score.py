from .metric import Metric

class AgreementScore(Metric):
    """
    Calculates agreement score between two sets of answers.

    AgreementScore computes the proportion of questions that received 
    consistent answers between a source (e.g., document) and a summary.
    """

    @staticmethod
    def _compute_metric(answers_src, answers_sum):
        """
        Computes the number of matches between the answers from source and summary.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.

        Returns:
            int: Number of questions with consistent answers.
        """
        answers_src_ls = list(answers_src.values())
        answers_sum_ls = list(answers_sum.values())
        n_matches = sum(1 for ans_src, ans_sum in zip(answers_src_ls, answers_sum_ls)
                        if ans_src.strip().lower() == ans_sum.strip().lower())
        return n_matches

    @staticmethod
    def compute(answers_src, answers_sum, n_questions):
        """
        Computes the agreement score.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.
            n_questions (int): Total number of questions.

        Returns:
            float: Agreement score.
        """
        n_matches = AgreementScore._compute_metric(answers_src, answers_sum)
        aggr_score = n_matches / n_questions
        return aggr_score