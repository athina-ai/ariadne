from .metric import Metric

class Noninformativeness(Metric):
    """
    Calculates noninformativeness score between two sets of answers.

    Noninformativeness computes the proportion of questions that received 
    received a 'Unknown' summary-based answer and an 'Yes/No' document-based answer.
    """

    @staticmethod
    def _compute_metric(answers_src, answers_sum):
        """
        Computes the number of questions with 'Unknown' summary-based answer and an 'Yes/No' document-based answer.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.

        Returns:
            int: Number of questions with questions with 'Unknown' summary-based answer and an 'Yes/No' document-based answer.
        """
        answers_src_ls = list(answers_src.values())
        answers_sum_ls = list(answers_sum.values())
        n_non_informativeness = 0 
        for ans_src, ans_sum in zip(answers_src_ls, answers_sum_ls):
            if ans_sum.strip().lower() == 'unknown' and ans_src.strip().lower() in ['yes', 'no']:
                n_non_informativeness += 1
        return n_non_informativeness

    @staticmethod
    def compute(answers_src, answers_sum, n_questions):
        """
        Computes the non-informativeness score.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.
            n_questions (int): Total number of questions.

        Returns:
            float: Non-informativeness score.
        """
        n_non_informativeness = Noninformativeness._compute_metric(answers_src, answers_sum)
        noninfo_score = n_non_informativeness / n_questions
        return noninfo_score