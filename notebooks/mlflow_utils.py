import numpy as np
import mlflow
from typing import Dict
import matplotlib.pyplot as plt
from loguru import logger

def plot_image(image: np.ndarray, image_name: str, plot_dir: str = None) -> str:
    """
    Plots the precision-recall curve.

    Args:
        y_true: Array of true y values
        y_pred: Array of predicted y values
        model_name: Name of model
        plot_dir: Directory to save plot in

    Returns:
        Ø
        Output path of plot
    """
    plt.figure(figsize=(15, 5))
    plt.gca().invert_xaxis()
    plt.legend()
    plt.imshow(image)
    plt.title('{}'.format(image_name))

    # Save figure
    if plot_dir:
        output_path = '{}/plots/image_{}.png'.format(plot_dir, image_name)
        plt.savefig(output_path)
        logger.info('Image saved to: {}'.format(output_path))
        return output_path


def log_mlflow(run_params: Dict, feature_name : str, mean : float, snapshot: np.ndarray) -> None:
    """
    Logs seg process

    Args:
        run_params: Dictionary containing parameters of run.
                    Expects keys for 'experiment', 'artifact_dir', 'iteration', and 'workspace_name'.
        
    Returns:
        None
    """
    mlflow.set_experiment(run_params['experiment'])

    #auc, recall, precision, f1 = evaluate_binary(y_true, y_pred)
    #roc_path = plot_roc(y_true, y_pred, '{} (auc = {:.2f})'.format(model_name, auc), run_params['artifact_dir'])
    #model_path = save_model(model, model_name, run_params['artifact_dir'])
    
    image_dir = plot_image(snapshot, feature_name, run_params['artifact_dir'])
    
    with mlflow.start_run(run_name=run_params['iteration']):
        mlflow.log_param('workspace_name', run_params['workspace_name'])
        mlflow.log_param('feature_name', feature_name)
        mlflow.log_metric('mean', mean)
        mlflow.log_artifact(image_dir)