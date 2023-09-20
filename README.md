## Overview
AriadneAI is an open-source library for evaluating LLM applications on text summarization. 


## Installation

To install, simply download the library and include it in your project.

```bash
pip install ariadne-ai
```

## Usage
Here's a simple usage example to load a json file for text summarization.
```python
loader = TextSummarizationLoader(format = 'json')
loader.load("path_to_your_file.json")
text_summarization_evaluator = SummarizationHallucinationEvaluator(text_summarization_loader)
text_summarization_evaluator.run()
```
or run

```python
 poetry run python main.py
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
