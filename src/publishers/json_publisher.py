from .publisher import Publisher
import json
import pandas as pd

class JSONPublisher(Publisher):
    """
    Publishes data in JSON format.

    Attributes:
        filename (str): The default output filename for the publisher.
    """

    def __init__(self, filename: str):
        """Initializes the JSONPublisher with a specified output filename."""
        self.filename = filename

    def write(self, data: dict):
        """ Writes data to a JSON file. """
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=4)

    def convert_pd_to_json(self, input_filename: str, output_filename: str):
        """ Converts data from a CSV file to a JSON format and saves it."""
        df = pd.read_csv(input_filename)
        data_to_save = df.to_dict('records')
        
        # Use the write method to save the data to the output file
        self.write(data_to_save)
