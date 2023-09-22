## Overview
Ariadne AI is an open-source library for evaluating LLM applications. The goal of the library is to provide LLM evaluators for RAG chatbots, text summarization, and LLM agents. Each evaluator is paired with metrics that help developers evaluate their LLMs and detect the reason for their failure cases. If labeled data available, machine learning metrics such as accuracy, recall, and precision can be also computed. The library also supports any evaluations for specialized tasks such as sentiment detection and PII detection within any LLM input and output. 


## Installation

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
 poetry run python example.py
```


## Text Summarization

![Text Summarization QAG Approach](docs/QAG_approach.png)

Our approach is a question-answer generation (QAG) framework, which allows us to pinpoint failure cases in production without the necessity for human-annotated reference summaries. 

Here is a breakdown of our approach:

1. **Question Generation:** The LLM formulates closed-ended (Yes/No) questions drawing from both the summary and the main document.
2. **Summary-based Answers:** An LLM answerer generator responds to these questions using only the summary as a reference. The potential responses include "Yes," "No," and "Unknown."
3. **Document-based Answers:** Similarly, the LLM answerer generator answers the same set of questions, but this time, it references the primary document. Possible responses remain "Yes," "No," and "Unknown."
4. **Evaluation Metrics:** The evaluation metrics assessing the consistency between the summary-based and document-based summaries are computed to draw conclusions.

To detect the type of failure cases, we compute the following evaluation metrics:

*  **Hallucination Score**: This metric captures the percentage of questions that received a 'Yes/No' summary-based answer and an 'Unknown' document-based answer. A high score suggests the summary might include content absent from the original document.

*  **Contradiction Score:**  This metric captures the percentage of questions that received  a 'Yes' summary-based answer and a 'No document-based answer, and vice-versa. A high score suggests the summary might include content that contradicts  the original document.

*  **Non-informativeness Score:**  This metric captures the percentage of questions that received a 'Unknown' summary-based answer and an 'Yes/No' document-based answer. A high score indicates that the summary may miss details from the  document or be very generic.

## Contribution 

Please feel free to reach out to christos@athina.ai or shiv@athina.ai if you would like to contribute. You could find more on how you could integrate the evaluations in your product here: https://docs.athina.ai.


## License

[Apache License 2.0](LICENSE)
