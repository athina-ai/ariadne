from .publisher import Publisher
import json
import pandas as pd

class PublisherLog(Publisher):
    """
    A class to publish log data in various formats.

    Attributes:
        filename (str): The output filename for the publisher.
        format (str): The format in which the data will be published, e.g., 'json', 'csv'.
    """

    def __init__(self, filename: str, format: str):
        """ Initializes the PublisherLog with an output filename and format."""
        self.filename = filename
        self.format = format

    def write(self, data: dict):
        """ Writes data in the specified format. """
        if self.format == 'json':
            self.write_json(data)
        elif self.format == 'csv':
            self.write_csv(data)
        else:
            raise NotImplementedError(f"The '{self.format}' format has not been implemented yet.")

    def write_json(self, data: dict):
        """ Writes data to a JSON file. """
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=4)

    def write_csv(self, data: dict):
        """ Writes data to a CSV file. """
        df = pd.DataFrame(data)
        df.to_csv(self.filename, index=False)

    def write_magik(self, data: dict):
        """ Writes data to a magik dashboard. """
        raise NotImplementedError("This method has not been implemented yet.")
