import os
import time
import mlflow
import yaml
from dotenv import load_dotenv
from .base_logger import BaseLogger
from ppocr.utils.logging import get_logger

# Load environment variables from .env file
load_dotenv()

class MLflowLogger(BaseLogger):
    def __init__(self, save_dir=None, config=None, mlflow_log_every_n_iter=1, name="default_run_name", **kwargs):
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self._run = None
        self.mlflow_log_every_n_iter = mlflow_log_every_n_iter
        self.logger = get_logger()

        # Fetch values from environment variables (or fallback to default)
        self.project = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")
        self.name = name

        # Set up MLflow experiment and start the run
        mlflow.set_experiment(self.project)
        _ = self.run

        # If there's a configuration, log it
        if self.config:
            self.log_config()

    @property
    def run(self):
        if self._run is None:
            self._run = mlflow.start_run(run_name=self.name)
        return self._run

    def log_metrics(self, metrics, prefix=None, step=None):
        if prefix == 'TRAIN' and step % self.mlflow_log_every_n_iter != 0:
            return
        if not prefix:
            prefix = ""
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}
        mlflow.log_metrics(updated_metrics, step=step, run_id=self.run.info.run_id) 

    def log_model(self, is_best, prefix, metadata=None):
        model_dir = os.path.join(self.save_dir, "..")
        for ext in [".pdparams", ".pdopt", ".states"]:
            model_path = os.path.join(model_dir, prefix + ext)
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path=f"weights/{prefix}", run_id=self.run.info.run_id)
        # Log metadata
        if metadata:
            mlflow.set_tags(metadata)  # Batch tags if possible

    def log_config(self):
        def flatten_dict(d, parent_key='', sep='.'):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        flattened_config = flatten_dict(self.config)
        mlflow.log_params(flattened_config, run_id=self.run.info.run_id)  # Batch parameters

        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)
        mlflow.log_artifact(config_path, artifact_path="configs", run_id=self.run.info.run_id)

    def close(self):
        """Finish the MLflow run."""
        mlflow.end_run()
