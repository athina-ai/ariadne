from ariadne_ai.loaders.summarization_loader import SummarizationLoader
from ariadne_ai.evaluators.summarization_hallucination_evaluator import SummarizationHallucinationEvaluator
from ariadne_ai.evaluators.summarization_informativeness_evaluator import SummarizationInformativenessEvaluator
# Constants and Configurations

# Path to the input file containing raw data.
INPUT_FILEPATH = 'data/text_summarization/xsum_sample.json'

# OpenAI API key (should be kept confidential).
OPEN_AI_KEY = None

# Each configuration specifies the log, performance file paths, model, and number of questions.
# Configurations for different runs
RUN_CONFIGS = [
    {
        'log_filepath': 'data/logs/log_sum_hal_eval_gpt_35_questions_5.json',
        'perf_filepath': 'data/logs/perf_sum_hal_eval_gpt_35_questions_5.txt',
        'llm_model': 'gpt-3.5-turbo-16k',
        'n_questions': 5
    },
    {
        'log_filepath': 'data/logs/log_sum_hal_eval_gpt_35_questions_2.json',
        'perf_filepath': 'data/logs/perf_sum_hal_eval_gpt_35_questions_2.txt',
        'llm_model': 'gpt-3.5-turbo-16k',
        'n_questions': 2
    },
    {
        'log_filepath': 'data/logs/log_sum_hal_eval_gpt_4_questions_5.json',
        'perf_filepath': 'data/logs/perf_sum_hal_eval_gpt_4_questions_5.txt',
        'llm_model': 'gpt-4',
        'n_questions': 5
    }
]

def initialize_loader() -> SummarizationLoader:
    """
    Initialize the summarization loader, load the data, and return the loader object.
    """
    loader = SummarizationLoader(
        col_document='document', 
        col_summary='summary', 
        col_label='hallucination_type',
        col_comment='hallucinated_span'
    )
    loader.load(INPUT_FILEPATH)
    return loader

def run_evaluation_hallucination(loader: SummarizationLoader, config: dict) -> None:
    """
    Given a loader and configuration, initialize an evaluator and run the hallucination evaluation.
    """
    evaluator = SummarizationHallucinationEvaluator(
        loader,
        log_filepath=config['log_filepath'],
        llm_model=config['llm_model'],
        performance_filepath=config['perf_filepath'],
        open_ai_key=OPEN_AI_KEY,
        n_questions=config.get('n_questions')
    )
    evaluator.run()

def run_evaluation_informativeness(loader: SummarizationLoader, config: dict) -> None:
    """
    Given a loader and configuration, initialize an evaluator and run the non-informativeness evaluation.
    """
    evaluator = SummarizationInformativenessEvaluator(
        loader,
        log_filepath=config['log_filepath'],
        llm_model=config['llm_model'],
        performance_filepath=config['perf_filepath'],
        open_ai_key=OPEN_AI_KEY,
        n_questions=config.get('n_questions')
    )
    evaluator.run()

def main():
    """
    Main execution function. Initializes the loader and runs evaluations for each configuration.
    """
    loader = initialize_loader()
    for config in RUN_CONFIGS:
        #run_evaluation_informativeness(loader, config)
        run_evaluation_hallucination(loader, config)

# Ensure that the main execution only occurs if this script is run directly (not imported).
if __name__ == "__main__":
    main()
