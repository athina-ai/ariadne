from ariadne_ai.loaders.rag_loader import RagLoader
from ariadne_ai.evaluators.rag.faithfulness_evaluator import FaithfulnessEvaluator
from ariadne_ai.evaluators.rag.context_relevance_evaluator import (
    ContextRelevanceEvaluator,
)
from ariadne_ai.evaluators.rag.answer_relevance_evaluator import (
    AnswerRelevanceEvaluator,
)

# Constants and Configurations

# Path to the input file containing raw data.
INPUT_FILEPATH = "data/rag/faithfulness_sample.json"
LOG_FILEPATH = "data/logs/log_rag_faith_eval.json"
PERF_FILEPATH = "data/logs/perf_rag_faith_eval.txt"

# OpenAI API key (should be kept confidential).
OPEN_AI_KEY = None

loader = RagLoader(
    col_question="question",
    col_context="context",
    col_answer="answer",
    col_label="label",
)

# Faithfulness
loader.load(INPUT_FILEPATH)
evaluator = FaithfulnessEvaluator(loader)
evaluator.run()

# Context Relevance
INPUT_FILEPATH = "data/rag/context_relevance_sample.json"
loader = RagLoader(
    col_question="question",
    col_context="context",
    col_answer="answer",
    col_label="label",
)
loader.load(INPUT_FILEPATH)
evaluator = ContextRelevanceEvaluator(loader=loader)
evaluator.run()

# Answer Relevance

INPUT_FILEPATH = "data/rag/answer_relevance_sample.json"
loader = RagLoader(
    col_question="question",
    col_context="context",
    col_answer="answer",
    col_label="label",
)
loader.load(INPUT_FILEPATH)
evaluator = AnswerRelevanceEvaluator(loader)
evaluator.run()
