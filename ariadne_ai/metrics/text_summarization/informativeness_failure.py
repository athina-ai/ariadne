from ..metric import Metric

class InformativenessFailure(Metric):
    """
    Calculates noninformativeness score between two sets of answers.

    Informativeness computes the proportion of questions that received 
    received a 'Unknown' summary-based answer and an 'Yes/No' document-based answer.
    """

    @staticmethod
    def _compute_metric(answers_src, answers_sum, questions):
        """
        Computes the number of questions with 'Unknown' summary-based answer and an 'Yes/No' document-based answer.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.

        Returns:
            int: Number of questions with questions with 'Unknown' summary-based answer and an 'Yes/No' document-based answer.
        """
        # get answers
        answers_src_ls = list(answers_src.values())
        answers_sum_ls = list(answers_sum.values())
        # init 
        non_info_questions =[]
        n_non_informativeness = 0 

        for idx, (ans_src, ans_sum) in enumerate(zip(answers_src_ls, answers_sum_ls)):
            if ans_sum.strip().lower() == 'unknown' and ans_src.strip().lower() in ['yes', 'no']:
                n_non_informativeness += 1
                non_info_question = questions[f"question {idx+1}"]
                non_info_questions.append(f'{non_info_question}')

        return n_non_informativeness, non_info_questions

    @staticmethod
    def compute(answers_src, answers_sum, questions, n_questions):
        """
        Computes the non-informativeness score.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.
            n_questions (int): Total number of questions.

        Returns:
            float: Non-informativeness score.
        """
        n_non_informativeness, non_info_questions = InformativenessFailure._compute_metric(answers_src, answers_sum, questions)
        is_informativeness_failure = 1 if n_non_informativeness > 0 else 0 
        non_info_score = n_non_informativeness / n_questions
        explanation = non_info_questions
        return is_informativeness_failure, explanation, non_info_score
    


