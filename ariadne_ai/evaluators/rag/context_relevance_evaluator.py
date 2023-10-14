from .rag_evaluator import RagEvaluator
from ...loaders.rag_loader import RagLoader
from ...metrics.rag.context_relevance_failure import ContextRelevanceFailure
from ...publishers.publisher_log import PublisherLog
from ...llms.rag.context_relevance import ContextRelevance
from typing import Optional


class ContextRelevanceEvaluator(RagEvaluator):
    """
    Evaluator for context relevance in rag chatbot.

    Attributes:
        dataset: Dataset containing instances for evaluation.
        context_relevance_evaluator: Evaluator for context relevance.
        publisher_log: JSON publisher to save the evaluation logs.
        performance_report_filename: txt file to save the perfrormance of a batch
        metrics: List of metrics to evaluate.
        logs: List to accumulate evaluation results for each instance.
    """

    # Chamge metric
    metric_str_to_class = {"context_relevance_failure": ContextRelevanceFailure}

    def __init__(
        self,
        loader,
        log_filepath="data/logs/log_rag_cont_rel_eval.json",
        log_format="json",
        performance_filepath="data/logs/perf_rag_cont_rel_eval.txt",
        llm_model="gpt-3.5-turbo",
        metrics=["context_relevance_failure"],
        open_ai_key=None,
        athina_api_key: Optional[str] = None,
        metadata: Optional[dict] = None,
        additional_instructions: Optional[str] = None,
    ):
        """
        Initialize the evaluator with given parameters.

        Args:
        - loader: An instance of SummarizationLoader.
        - log_filepath: Path to save the logs.
        - llm_model: Language model to be used.
        - metrics: List of metrics for evaluation.
        """
        if not isinstance(loader, RagLoader):
            raise TypeError("Loader must be an instance of RagLoader")
        # Load data
        self.dataset = loader.processed_dataset
        # Intialize LLMs
        self.llm_model = llm_model
        self.context_relevance_evaluator = ContextRelevance(
            llm_model,
            open_ai_key,
            athina_api_key=athina_api_key,
            metadata=metadata,
            additional_instructions=additional_instructions,
        )
        # Initialize logging
        self.log_format = log_format
        if log_format is not None:
            self.publisher_log = PublisherLog(log_filepath, log_format)
        self.logs = []
        self.n_instances = 0
        # Intialize metrics
        self.performance_filepath = performance_filepath
        self.metrics = metrics
        self.label_counts = {}
        for metric in metrics:
            setattr(self, f"{metric}_scores", {})

    def _evaluate_element(self, instance):
        """Evaluate an instance for hallucination."""
        question = instance["question"]
        context = instance["context"]
        if "label" in instance:
            label = instance["label"]
        else:
            label = "overall"
        # Run LLM Evaluator for faithfulness
        cont_rel_eval = self.context_relevance_evaluator.evaluate(question, context)

        metric_results = {}
        # Compute metrics
        if cont_rel_eval is None:
            metric_results["evaluation"] = "undefined"
        else:
            for metric in self.metrics:
                metric_class = self.metric_str_to_class.get(metric)
                metric_result, explanation = metric_class.compute(cont_rel_eval)
                metric_results[metric] = metric_result
                metric_results["reason"] = explanation
                self.update_metric_aggr(metric, label, metric_result)
            self.n_instances = self.n_instances + 1
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
        return {
            "question": question,
            "context": context,
            "label": label,
            **metric_results,
        }

    def update_metric_aggr(self, metric, label, aggr_score):
        """Update the aggregated score for a specific metric and label."""
        metric_aggr = getattr(self, f"{metric}_scores", {})
        current_score = metric_aggr.get(label, 0)
        metric_aggr[label] = current_score + aggr_score
        setattr(self, f"{metric}_scores", metric_aggr)

    def get_metric_aggr(self, metric, label):
        """Compute the average scores based on the provided score dictionary."""
        metric_aggr = getattr(self, f"{metric}_scores", {})
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
        avg_scores["overall"] = sum_score / n_instances
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
        with open(filename, "w") as f:
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
        if self.log_format is not None:
            self.publisher_log.write(self.logs)
        return self.logs
