## Overview
AriadneAI is an open-source library for evaluating LLM applications on text summarization. 


## Installation

To install, simply download the library and include it in your project.

```bash
pip install ariadneai
```

## Usage
Here's a simple usage example to load a json file for text summarization.
```python
loader = TextSummarizationLoader()
loader.load_json("path_to_your_file.json")
print(loader.processed_dataset)
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
