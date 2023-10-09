from .publisher import Publisher
import json
import os


class PublisherLog(Publisher):
    """
    A class to publish log data in various formats.

    Attributes:
        filename (str): The output filename for the publisher.
        format (str): The format in which the data will be published, e.g., 'json', 'csv'.
    """

    def __init__(self, filename: str, format: str):
        """Initializes the PublisherLog with an output filename and format."""
        self.filename = filename
        self.format = format
        directory_path = os.path.dirname(self.filename)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def write(self, data: dict):
        """Writes data in the specified format."""
        if self.format == "json":
            self.write_json(data)
        elif self.format == "csv":
            self.write_csv(data)
        else:
            raise NotImplementedError(
                f"The '{self.format}' format has not been implemented yet."
            )

    def write_json(self, data: dict):
        """Writes data to a JSON file."""
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)

    def write_csv(self, data: dict):
        """Writes data to a CSV file."""
        pass
