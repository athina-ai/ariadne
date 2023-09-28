from ariadne_ai.loaders.rag_generation_loader import RagGenerationLoader
from ariadne_ai.evaluators.rag_faithfulness_evaluator import RagFaithfulnessEvaluator
#from ariadne_ai.evaluators.summarization_hallucination_evaluator import SummarizationHallucinationEvaluator
#from ariadne_ai.evaluators.summarization_informativeness_evaluator import SummarizationInformativenessEvaluator
# Constants and Configurations

# Path to the input file containing raw data.
INPUT_FILEPATH = 'data/rag/faithfulness_sample.json'
LOG_FILEPATH = 'data/logs/log_rag_faith_eval.json'
PERF_FILEPATH = 'data/logs/perf_rag_faith_eval.txt'

# OpenAI API key (should be kept confidential).
OPEN_AI_KEY = None

loader = RagGenerationLoader(col_context='context', col_answer= 'answer', col_label ='label')
loader.load(INPUT_FILEPATH)


evaluator = RagFaithfulnessEvaluator(loader)
evaluator.run()


