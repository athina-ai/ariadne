

from src.loaders.text_summarization_loader import TextSummarizationLoader

input_filepath = 'data/text_summarization/xsum_sample.json'

text_summarization_loader = TextSummarizationLoader(col_document = 'document', col_summary = 'summary',  col_label = 'hallucination_type', col_comment = 'hallucinated_span')

text_summarization_loader.load_json(input_filepath)

print(text_summarization_loader.processed_dataset)
