from .summarization_evaluator import SummarizationEvaluator
from ..loaders.summarization_loader import SummarizationLoader
from ..metrics.aggreement_score import AgreementScore 
from ..metrics.contradiction_score import ContradictionScore
from ..metrics.hallucination_score import HallucinationScore 
from ..publishers.publisher_log import PublisherLog
from ..llms.question_generator import QuestionGenerator
from ..llms.question_answerer import QuestionAnswerer

class SummarizationHallucinationEvaluator(SummarizationEvaluator):
    """
    Evaluator for hallucinations in text summarizations. 

    Attributes:
        dataset: Dataset containing instances for evaluation.
        question_generator: Question generator based on summaries.
        question_answerer: Question answerer for the questions based on documents/summaries.
        publisher_log: JSON publisher to save the evaluation logs.
        metrics: List of metrics to evaluate.
        logs: List to accumulate evaluation results for each instance.
        n_questions: Number of questions to be generated for each summary.
    """

    metric_str_to_class = {
        'agreement_score': AgreementScore,
        'hallucination_score': HallucinationScore,
        'contradiction_score': ContradictionScore
    }

    def __init__(self, loader, log_filepath='data/logs/log_sum_hal_eval.json', log_format = 'json', n_questions=5, 
                 llm_model='gpt-3.5-turbo', metrics=['agreement_score', 'hallucination_score', 'contradiction_score'], open_ai_key = None):
        """
        Initialize the evaluator with given parameters.

        Args:
        - loader: An instance of SummarizationLoader.
        - log_filepath: Path to save the logs.
        - n_questions: Number of questions to generate for summaries.
        - llm_model: Language model to be used.
        - metrics: List of metrics for evaluation.
        """
        if not isinstance(loader, SummarizationLoader):
            raise TypeError("Loader must be an instance of SummarizationLoader")
        
        # Load data
        self.dataset = loader.processed_dataset
        # Intialize LLMs
        self.n_questions = n_questions
        self.question_generator = QuestionGenerator(llm_model, n_questions,  open_ai_key)
        self.question_answerer = QuestionAnswerer(llm_model, open_ai_key)
        # Initialize logging
        self.publisher_log = PublisherLog(log_filepath, log_format)

        self.logs = []
        self.metrics = metrics

    def _evaluate_element(self, instance):
        """Evaluate an instance for hallucination."""
        document = instance['document']
        summary = instance['summary']
        
        # Generate questions based on summary
        questions = self.question_generator.generate(summary)
        
        # Get answers from document and summary
        answers_doc = self.question_answerer.answer(questions, document)
        answers_sum = self.question_answerer.answer(questions, summary)
        
        # Compute metrics
        metric_results = {}
        for score in self.metrics:
            metric_class = self.metric_str_to_class.get(score)
            metric_result = metric_class.compute(answers_doc, answers_sum, self.n_questions)
            metric_results[score] = metric_result
        
        return {
            'document': document,
            'summary': summary,
            'questions': questions,
            'answers_doc': answers_doc,
            'answers_sum': answers_sum,
            'hallucination_type': instance['label'],
            **metric_results
        }

    def run(self):
        """Evaluate all instances in the dataset."""
        for instance in self.dataset:
            log = self._evaluate_element(instance)
            self.logs.append(log)
        self.publisher_log.write(self.logs)