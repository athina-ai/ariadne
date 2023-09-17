from src.loaders.summarization_loader import SummarizationLoader
from src.evaluators.summarization_hallucination_evaluator import SummarizationHallucinationEvaluator

# Define the input filepath
input_filepath = 'data/text_summarization/xsum_sample.json'

# Initialize a summarization loader 
text_summarization_loader = SummarizationLoader(
    col_document='document', 
    col_summary='summary', 
    col_label='hallucination_type', 
    col_comment='hallucinated_span'
)

# Load data from the specified JSON file
text_summarization_loader.load_json(input_filepath)

# Create an evaluator object using the loaded data
text_summarization_evaluator = SummarizationHallucinationEvaluator(text_summarization_loader)

# Run the evaluation
text_summarization_evaluator.run()
