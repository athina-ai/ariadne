from .loader import Loader
import json

class SummarizationLoader(Loader):
    """
    A data loader for text summarization datasets.


    Attributes:
        col_document (str): Column name corresponding to the main document text.
        col_summary (str): Column name corresponding to the summary of the document.
        col_label (str, optional): Column name corresponding to labels, if any.
        col_comment (str, optional): Column name corresponding to additional comments, if any.
        raw_dataset (dict): Raw dataset as loaded from the source.
        processed_dataset (list): Processed dataset with documents, summaries, and other attributes if present.
    """
    
    def __init__(self, col_document='document', col_summary='summary', 
                 col_label=None, col_comment=None, format = 'json'):
        """ Initializes the loader with specified or default column names."""
        self.col_document = col_document
        self.col_summary = col_summary
        self.col_label = col_label
        self.col_comment = col_comment
        self._raw_dataset = {}
        self._processed_dataset = []
        self.format = format

    @property
    def processed_dataset(self):
        """ Returns the processed dataset."""
        return self._processed_dataset
    
    @property
    def raw_dataset(self):
        """ Returns the raw dataset."""
        return self._raw_dataset

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes
        
        Raises:
            KeyError: If mandatory columns (document or summary) are missing in the raw dataset.
            KeyError: If optional columns (label or comment) are missing in the raw dataset if defined.
        """
        for raw_instance in self._raw_dataset:
            # Check for mandatory columns in raw_instance
            if self.col_document not in raw_instance:
                raise KeyError(f"'{self.col_document}' not found in provided data.")
            if self.col_summary not in raw_instance:
                raise KeyError(f"'{self.col_summary}' not found in provided data.")
            # Create a processed instance with mandatory fields
            processed_instance = {
                'document': raw_instance[self.col_document],
                'summary': raw_instance[self.col_summary]
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
        """Loads and processes data from a list of dictionaries."""
        self._raw_dataset = data
        self.process()

    def load_response(self, document, summary, comment = None, label = None) -> None:
        """Loads and processes a response of text summarization"""
        raw_instance = {'document': document, 'summary': summary}
        if comment is not None:
            raw_instance['comment'] =comment
        if label is not None:
            raw_instance['label'] =comment
        self._raw_dataset = [raw_instance]
        self.process()

    def load(self, data: list) -> None:
        """" """
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

   
