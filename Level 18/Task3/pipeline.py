from zenml import pipeline
from steps.load_data import load_digits_data
from steps.clean_data import clean_data


@pipeline
def data_cleaning_pipeline():
    """
    A pipeline that loads data and performs data cleaning.
    
    Steps:
    1. Load digits dataset
    2. Clean the data (drop columns, fill nulls)
    """
    # Load the data
    X, y = load_digits_data()
    
    # Clean the data
    cleaned_df, targets = clean_data(X, y)
    
    return cleaned_df, targets


if __name__ == "__main__":
    print("Starting ZenML data cleaning pipeline...")
    pipeline_run = data_cleaning_pipeline()
    print("Pipeline execution completed!")
    print(f"Pipeline run ID: {pipeline_run.id}")
