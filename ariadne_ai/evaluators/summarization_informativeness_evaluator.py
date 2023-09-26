from .summarization_evaluator import SummarizationEvaluator
from ..loaders.summarization_loader import SummarizationLoader
from ..metrics.aggreement_score import AgreementScore 
from ..metrics.noninformativeness_score import Noninformativeness
from ..publishers.publisher_log import PublisherLog
from ..llms.question_generator import QuestionGenerator
from ..llms.question_answerer import QuestionAnswerer

class SummarizationInformativenessEvaluator(SummarizationEvaluator):
    """
    Evaluator for informativeness in text summarizations. 

    Attributes:
        dataset: Dataset containing instances for evaluation.
        question_generator: Question generator based on documents.
        question_answerer: Question answerer for the questions based on documents/summaries.
        publisher_log: JSON publisher to save the evaluation logs.
        performance_report_filename: txt file to save the perfrormance of a batch
        metrics: List of metrics to evaluate.
        logs: List to accumulate evaluation results for each instance.
        n_questions: Number of questions to be generated for each summary.
    """

    metric_str_to_class = {
        'agreement_score': AgreementScore,
        'non_informativeness_score': Noninformativeness
    }

    def __init__(self, loader, log_filepath='data/logs/log_sum_hal_eval.json', log_format = 'json', performance_filepath = 'data/logs/perf_sum_hal_eval.txt',  n_questions=5, 
                 llm_model='gpt-3.5-turbo', metrics=['agreement_score', 'non_informativeness_score'], open_ai_key = None):
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
        self.llm_model = llm_model
        self.n_questions = n_questions
        self.question_generator = QuestionGenerator(llm_model, n_questions,  open_ai_key)
        self.question_answerer = QuestionAnswerer(llm_model, open_ai_key)
        # Initialize logging
        self.publisher_log = PublisherLog(log_filepath, log_format)
        self.logs = []
        self.n_instances = 0 
        # Intialize metrics
        self.performance_filepath = performance_filepath
        self.metrics = metrics
        self.label_counts ={}
        for metric in metrics:
            setattr(self, f'{metric}_scores', {})

    def _evaluate_element(self, instance):
        """Evaluate an instance for non-informativeness."""
        document = instance['document']
        summary = instance['summary']
        if 'label' in instance:
            label = instance['label']
        else:
            label = 'overall'
        
        # Generate questions based on document
        questions = self.question_generator.generate(document)
        
        # Get answers from document and summary
        answers_doc = self.question_answerer.answer(questions, document)
        answers_sum = self.question_answerer.answer(questions, summary)
        metric_results = {}
        print(answers_doc, answers_sum, questions)
        # Compute metrics
        if( answers_doc is None or answers_sum is None or questions is None):
            metric_results['evaluation'] = 'undefined'
        else:
            for score in self.metrics:
                metric_class = self.metric_str_to_class.get(score)
                metric_result = metric_class.compute(answers_doc, answers_sum, self.n_questions)
                metric_results[score] = metric_result
                self.update_metric_aggr(score, label, metric_result)
            self.n_instances = self.n_instances +1
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
        return {
            'document': document,
            'summary': summary,
            'questions': questions,
            'answers_doc': answers_doc,
            'answers_sum': answers_sum,
            'label': label,
            **metric_results
        }

    def update_metric_aggr(self, metric, label, aggr_score):
        """ Update the aggregated score for a specific metric and label."""
        metric_aggr = getattr(self, f'{metric}_scores', {})
        current_score = metric_aggr.get(label, 0)
        metric_aggr[label] = current_score + aggr_score
        setattr(self, f'{metric}_scores', metric_aggr)

    def get_metric_aggr(self, metric, label):
        """Compute the average scores based on the provided score dictionary."""
        metric_aggr = getattr(self, f'{metric}_scores', {})
        return metric_aggr.get(label, None)
    
    def get_average_scores(self, score_dict):
        """Compute average scores for a metric"""
        avg_scores = {}
        sum_score = 0 
        n_instances = 0
        for label_type, total_score in score_dict.items(): 
            avg_scores[label_type] = total_score / self.label_counts[label_type]
            sum_score = sum_score + total_score
            n_instances = n_instances + self.label_counts[label_type]
        avg_scores['overall'] = sum_score/n_instances
        return avg_scores
    
    def compute_average_scores(self):
        """Compute average scores for each metric."""
        avg_scores = {}
        for metric in self.metrics:
            scores = getattr(self, f"{metric}_scores")
            avg_score = self.get_average_scores(scores)
            avg_scores[metric] = avg_score
        return avg_scores
        
    def generate_performance_report(self, filename):
        """Generate a performance report and save it to the provided filename."""
        avg_scores = self.compute_average_scores()
        with open(filename, 'w') as f:
            f.write(f"Number of Questions: {self.n_questions}\n")
            f.write(f"Model: {self.llm_model}\n")
            f.write("\nNumber of instances per Label:\n")
            for label, cnt in self.label_counts.items():
                f.write(f"{label}: {cnt}\n")
            f.write(f"total: {self.n_instances}\n")
            for metric in self.metrics:
                f.write(f"\nAverage {metric} per Label:\n")
                for label, avg in avg_scores[metric].items():
                    f.write(f"{label}: {avg}\n")



    def run(self):
        """Evaluate all instances in the dataset."""
        for instance in self.dataset:
            log = self._evaluate_element(instance)
            self.logs.append(log)
        self.generate_performance_report(self.performance_filepath)
        self.publisher_log.write(self.logs)

    



    