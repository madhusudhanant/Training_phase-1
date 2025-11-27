from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model


@pipeline
def model_evaluation_pipeline():
    """
    A pipeline that ingests data, trains a model, and evaluates its performance.
    """
    # Ingest the data
    data = ingest_data()

    # Train the model
    model = train_model(data)

    # Evaluate the model
    metrics = evaluate_model(model=model, data=data)
    
    return model, metrics


if __name__ == "__main__":
    print("Starting ZenML model evaluation pipeline...")
    pipeline_run = model_evaluation_pipeline()
    print("Pipeline execution completed!")
    print(f"Pipeline run ID: {pipeline_run.id}")
