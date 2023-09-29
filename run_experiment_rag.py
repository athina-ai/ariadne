from ariadne_ai.loaders.rag_generation_loader import RagGenerationLoader
from ariadne_ai.evaluators.rag_faithfulness_evaluator import RagFaithfulnessEvaluator
from ariadne_ai.evaluators.rag_context_relevance_evaluator import RagContextRelevanceEvaluator
from ariadne_ai.evaluators.rag_answer_relevance_evaluator import RagAnswerRelevanceEvaluator
# Constants and Configurations

# Path to the input file containing raw data.
INPUT_FILEPATH = 'data/rag/faithfulness_sample.json'
LOG_FILEPATH = 'data/logs/log_rag_faith_eval.json'
PERF_FILEPATH = 'data/logs/perf_rag_faith_eval.txt'

# OpenAI API key (should be kept confidential).
OPEN_AI_KEY = None

loader = RagGenerationLoader(
    col_question= 'question', 
    col_context='context', 
    col_answer= 'answer', 
    col_label ='label')
loader.load(INPUT_FILEPATH)

# Faithfullness
evaluator = RagFaithfulnessEvaluator(loader)
evaluator.run()

# Context Relevance
INPUT_FILEPATH = 'data/rag/context_relevance_sample.json'

loader = RagGenerationLoader(
    col_question= 'question', 
    col_context='context', 
    col_answer= 'answer', 
    col_label ='label')
loader.load(INPUT_FILEPATH)

evaluator = RagContextRelevanceEvaluator(loader)
evaluator.run()

# Answer Relevance

INPUT_FILEPATH = 'data/rag/answer_relevance_sample.json'

loader = RagGenerationLoader(
    col_question= 'question', 
    col_context='context', 
    col_answer= 'answer', 
    col_label ='label')
loader.load(INPUT_FILEPATH)

evaluator = RagAnswerRelevanceEvaluator(loader)
evaluator.run()

