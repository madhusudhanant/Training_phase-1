from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model


@pipeline
def model_training_pipeline():
    """
    A pipeline that ingests data and trains a model.
    """
    # Ingest the data
    data = ingest_data()
    
    # Train the model
    model = train_model(data)
    
    return model


if __name__ == "__main__":
    print("Starting ZenML model training pipeline...")
    pipeline_run = model_training_pipeline()
    print("Pipeline execution completed!")
    print(f"Pipeline run ID: {pipeline_run.id}")
