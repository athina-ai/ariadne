from ariadne_ai.loaders.summarization_loader import SummarizationLoader
from ariadne_ai.evaluators.summarization_hallucination_evaluator import SummarizationHallucinationEvaluator

# Define the input filepath
input_filepath = 'data/text_summarization/xsum_sample.json'

# Define the output filepath
output_filepath = 'data/logs/log_sum_hal_eval.json'

#Define your OpenAI key
open_ai_key = None

# Initialize a summarization loader 
text_summarization_loader = SummarizationLoader(
    col_document='document', 
    col_summary='summary', 
    col_label='hallucination_type', 
    col_comment='hallucinated_span'
)

# Load data from the specified JSON file
text_summarization_loader.load(input_filepath)

# Create an evaluator object using the loaded data
text_summarization_evaluator = SummarizationHallucinationEvaluator(text_summarization_loader, output_filepath, open_ai_key=open_ai_key )

# Run the evaluation
text_summarization_evaluator.run()
