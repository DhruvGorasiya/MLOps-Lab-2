import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import pickle
import os
import numpy as np
import json
from datetime import datetime


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    
    return serialized_data
    

def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized clustered data.
    """
    df = pickle.loads(data)
    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return clustering_serialized_data


def build_save_model(data, filename):
    """
    Builds a KMeans clustering model, saves it to a file, and returns SSE values.

    Args:
        data (bytes): Serialized data for clustering.
        filename (str): Name of the file to save the clustering model.

    Returns:
        list: List of SSE (Sum of Squared Errors) values for different numbers of clusters.
    """
    df = pickle.loads(data)
    kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    # Create the model directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)

    # Save the trained model to a file
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans, f)
    return sse

def load_model_elbow(filename, sse):
    """
    Loads a saved KMeans clustering model and determines the number of clusters using the elbow method.

    Args:
        filename (str): Name of the file containing the saved clustering model.
        sse (list): List of SSE values for different numbers of clusters.

    Returns:
        str: A string indicating the predicted cluster and the number of clusters based on the elbow method.
    """
    
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    # Load the saved model from a file
    loaded_model = pickle.load(open(output_path, 'rb'))

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    
    kl = KneeLocator(
        range(1, 50), sse, curve="convex", direction="decreasing"
    )

    # Optimal clusters
    print(f"Optimal no. of clusters: {kl.elbow}")

    # Make predictions on the test data
    predictions = loaded_model.predict(df)
    
    return predictions[0]


def validate_data_quality(data):
    """
    Validates the quality of the input data and returns validation results.
    
    Args:
        data (bytes): Serialized data to be validated.
        
    Returns:
        dict: Dictionary containing validation results and statistics.
    """
    df = pickle.loads(data)
    
    validation_results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "validation_timestamp": datetime.now().isoformat()
    }
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        outliers_info[col] = outliers
    
    validation_results["outliers_count"] = outliers_info
    
    # Serialize and return results
    return pickle.dumps(validation_results)


def evaluate_model_performance(data, sse):
    """
    Evaluates the clustering model performance using multiple metrics.
    
    Args:
        data (bytes): Serialized preprocessed data.
        sse (list): List of SSE values for different numbers of clusters.
        
    Returns:
        dict: Dictionary containing model performance metrics.
    """
    df = pickle.loads(data)
    
    # Find optimal number of clusters using elbow method
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    
    if optimal_k is None:
        optimal_k = 3  # Default fallback
    
    # Train model with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df)
    
    # Calculate performance metrics
    silhouette_avg = silhouette_score(df, cluster_labels)
    
    # Calculate inertia (SSE) for optimal k
    optimal_sse = sse[optimal_k - 1] if optimal_k <= len(sse) else sse[-1]
    
    performance_metrics = {
        "optimal_clusters": optimal_k,
        "silhouette_score": float(silhouette_avg),
        "inertia": float(optimal_sse),
        "total_samples": len(df),
        "features_used": df.shape[1],
        "evaluation_timestamp": datetime.now().isoformat()
    }
    
    return pickle.dumps(performance_metrics)


def save_results_to_working_data(predictions, validation_results, performance_metrics):
    """
    Saves all results to the working_data directory for analysis.
    
    Args:
        predictions: Model predictions from load_model_elbow function.
        validation_results (bytes): Serialized validation results.
        performance_metrics (bytes): Serialized performance metrics.
        
    Returns:
        str: Path to the saved results file.
    """
    # Deserialize the data
    validation_data = pickle.loads(validation_results)
    performance_data = pickle.loads(performance_metrics)
    
    # Create working_data directory if it doesn't exist
    working_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "working_data")
    os.makedirs(working_dir, exist_ok=True)
    
    # Combine all results
    final_results = {
        "prediction": int(predictions),
        "data_validation": validation_data,
        "model_performance": performance_data,
        "processing_completed_at": datetime.now().isoformat()
    }
    
    # Save results as JSON for easy reading
    results_file = os.path.join(working_dir, f"ml_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    print(f"Model prediction: {predictions}")
    print(f"Optimal clusters: {performance_data.get('optimal_clusters', 'N/A')}")
    print(f"Silhouette score: {performance_data.get('silhouette_score', 'N/A'):.4f}")
    
    return results_file


def generate_data_summary(data):
    """
    Generates a comprehensive summary of the dataset.
    
    Args:
        data (bytes): Serialized data to summarize.
        
    Returns:
        dict: Dictionary containing data summary statistics.
    """
    df = pickle.loads(data)
    
    summary = {
        "dataset_info": {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "memory_usage": df.memory_usage(deep=True).sum()
        },
        "descriptive_stats": df.describe().to_dict(),
        "data_quality": {
            "missing_values_percent": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": df.duplicated().sum()
        },
        "generated_at": datetime.now().isoformat()
    }
    
    return pickle.dumps(summary)
