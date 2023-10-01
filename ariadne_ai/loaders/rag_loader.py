from .loader import Loader
import json

class RagLoader(Loader):
    """
    This class is a data loader for retrieval augmented generation (RAG) datasets.

    Attributes:
        col_question (str): The column name corresponding to the user's question.
        col_context (str): The column name corresponding to the retrieved context.
        col_answer (str): The column name corresponding to the answer.
        col_label (str, optional): The column name corresponding to labels, if any.
        col_comment (str, optional): The column name corresponding to additional comments, if any.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with queries, context, response and other attributes if present.
    """
    
    def __init__(self, col_question ='question',  col_context='context', col_answer = 'answer',
                 col_label=None, col_comment=None, format = 'json'):
        """ 
        Initializes the loader with specified or default column names.
        """
        self.col_question = col_question
        self.col_context = col_context
        self.col_answer = col_answer
        self.col_label = col_label
        self.col_comment = col_comment
        self._raw_dataset = {}
        self._processed_dataset = []
        self.format = format

    @property
    def processed_dataset(self):
        """ 
        Returns the processed dataset.
        """
        return self._processed_dataset
    
    @property
    def raw_dataset(self):
        """ 
        Returns the raw dataset.
        """
        return self._raw_dataset

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.
        
        Raises:
            KeyError: If mandatory columns (question, context or response) are missing in the raw dataset.
            KeyError: If optional columns (label or comment) are missing in the raw dataset if defined.
        """
        for raw_instance in self._raw_dataset:
            # Check for mandatory columns in raw_instance
            if self.col_question not in raw_instance:
                raise KeyError(f"'{self.col_question}' not found in provided data.")
            if self.col_context not in raw_instance:
                raise KeyError(f"'{self.col_context}' not found in provided data.")
            if self.col_answer not in raw_instance:
                raise KeyError(f"'{self.col_answer}' not found in provided data.")
            # Create a processed instance with mandatory fields
            processed_instance = {
                'question': raw_instance[self.col_question],
                'context': raw_instance[self.col_context],
                'answer': raw_instance[self.col_answer]
            }

           # Add optional attributes if they exist
            if self.col_label is not None and self.col_label in raw_instance:
                processed_instance['label'] = raw_instance[self.col_label]
            if self.col_comment is not None and self.col_comment in raw_instance:
                processed_instance['comment'] = raw_instance[self.col_comment]

            # Store the results
            self._processed_dataset.append(processed_instance)

    def load_json(self, filename: str) -> None:
        """
        Loads and processes data from a JSON file.
        
        Raises:
            FileNotFoundError: If the specified JSON file is not found.
            json.JSONDecodeError: If there's an issue decoding the JSON.
        """
        try:
            with open(filename, 'r') as f:
                self._raw_dataset = json.load(f)
                self.process()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON: {e}")
    
    def load_dict(self, data: list) -> None:
        """
        Loads and processes data from a list of dictionaries.
        """
        self._raw_dataset = data
        self.process()

    def load_response(self, question, context, answer, comment = None, label = None) -> None:
        """
        Loads and processes a response of text summarization.
        """
        raw_instance = {'question':question, 'context': context, 'answer': answer}
        if comment is not None:
            raw_instance['comment'] =comment
        if label is not None:
            raw_instance['label'] =comment
        self._raw_dataset = [raw_instance]
        self.process()

    def load(self, data: list) -> None:
        """
        Loads data based on the format specified.
        """
        if self.format == 'json':
            self.load_json(data)
        elif self.format == 'dict':
            self.load_dict(data)
        else:
            raise NotImplementedError("This file format has not been supported yet.")
    
        
    def load_csv(self) -> None:
        """
        Placeholder for loading data from a CSV file.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("This method has not been implemented yet.")

    def load_pandas(self) -> None:
        """
        Placeholder for loading data from a pandas DataFrame.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("This method has not been implemented yet.")

   

