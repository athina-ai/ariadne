
from src.loaders.summarization_loader import SummarizationLoader

from src.evaluators.summarization_hallucination_evaluator import SummarizationHallucinationEvaluator


input_filepath = 'data/text_summarization/xsum_sample.json'
text_summarization_loader = SummarizationLoader(col_document = 'document', col_summary = 'summary',  col_label = 'hallucination_type', col_comment = 'hallucinated_span')
text_summarization_loader.load_json(input_filepath)

text_summarization_evaluator = SummarizationHallucinationEvaluator(text_summarization_loader)
text_summarization_evaluator.run()