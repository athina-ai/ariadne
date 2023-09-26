import pytest
from src.loaders.summarization_loader import SummarizationLoader

def test_keyerror_on_missing_column():
    """ Missing "summary" key on purpose to trigger KeyError """
    test_data = [
        {"document": "doc1", "summary": "summary1"},
        {"document": "doc2"}  
    ]
    # Check if KeyError is raised 
    with pytest.raises(KeyError):
        SummarizationLoader(format ='dict').load(test_data)


def test_keyerror_on_missing_defining_columns():
    """ Missing defining document and summary columns in initialization on purpose to trigger KeyError """
    test_data = [{"doc": "doc1", "sum": "summary1"}]    
    text_summarization_loader = SummarizationLoader(format ='dict')
    # Check if KeyError is raised 
    with pytest.raises(KeyError):
        text_summarization_loader.load_dict(test_data)
    


def test_loader_renames_documnet_summary_columns():
    """ Checking if the loader renames the document and summary column correctly"""
    test_data = [{"doc": "doc1", "sum": "summary1"}]
    text_summarization_loader = SummarizationLoader(col_document='doc', col_summary='sum', format ='dict')
    text_summarization_loader.load(test_data)
    processed_data = text_summarization_loader.processed_dataset
    assert(processed_data[0]["document"] == test_data[0]["doc"] )
    assert(processed_data[0]["summary"] == test_data[0]["sum"] )



def test_loader_instance():
    text_summarization_loader = SummarizationLoader()
    text_summarization_loader.load_response(document= 'doc', summary='sum')
    processed_data = text_summarization_loader.processed_dataset
    assert(processed_data[0]["document"] == 'doc' )
    assert(processed_data[0]["summary"] == 'sum' )