import os
import re
import shutil
import mlflow
import paddle2onnx
import yaml
from dotenv import load_dotenv
from .base_logger import BaseLogger
from ppocr.utils.logging import get_logger
from ppocr.utils.export_model import export
from ppocr.utils.triton_flavor import log_model as triton_log_model

# Load environment variables from .env file
load_dotenv()

# Helper function to get or create an experiment, taken from https://github.com/mlflow/mlflow/issues/2111
def get_or_create_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)
    if not experiment:
        experiment_id = mlflow.create_experiment(name)
        experiment = mlflow.get_experiment(experiment_id)
    return experiment

class MLflowLogger(BaseLogger):
    def __init__(self, save_dir=None, config=None, mlflow_log_every_n_iter=1, name="default_run_name", **kwargs):
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self._run = None
        self.mlflow_log_every_n_iter = mlflow_log_every_n_iter
        self.logger = get_logger()

        # Fetch values from environment variables (or fallback to default)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.enable_system_metrics_logging()
        self.project = os.getenv("MLFLOW_EXPERIMENT_NAME", "test")
        self.name = name

        # Set up MLflow experiment and start the run
        experiment_id = get_or_create_experiment(self.project).experiment_id
        mlflow.set_experiment(experiment_id=experiment_id)

        # If there's a configuration, log it
        if self.config:
            self.log_config()

    @property
    def run(self):
        if self._run is None:
            run_name = self._create_run_name()
            self._run = mlflow.start_run(run_name=run_name)
        return self._run
    
    def _create_run_name(self):
        # Fetch existing run names
        current_runs = mlflow.search_runs(experiment_names=[self.project])['tags.mlflow.runName'].dropna().tolist()
        # Filter runs matching the exact pattern "name_{number}"
        pattern = re.compile(rf"^{re.escape(self.name)}_(\d+)$")
        indices = []
        for run in current_runs:
            match = pattern.match(run)
            if match:
                indices.append(int(match.group(1)))
        # Determine the next index
        next_index = max(indices, default=0) + 1
        return f"{self.name}_{next_index}"

    def log_metrics(self, metrics, prefix=None, step=None):
        if prefix == 'TRAIN' and step % self.mlflow_log_every_n_iter != 0:
            return
        if not prefix:
            prefix = ""
        updated_metrics = {prefix.lower() + "/" + k: v for k, v in metrics.items()}
        mlflow.log_metrics(updated_metrics, step=step, run_id=self.run.info.run_id) 

    def log_model(self, is_best, prefix, metadata=None):
        if not is_best:
            return

        model_dir = os.path.join(self.save_dir, "..")

        # Log paddle model
        for ext in [".pdparams", ".pdopt", ".states"]:
            model_path = os.path.join(model_dir, prefix + ext)
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path=f"paddle_models/{prefix}", run_id=self.run.info.run_id)

        # Log ONNX model
        # Firstly, temporarily export the model to an inference model
        inference_config = self.config.copy()
        inference_config["Global"]["pretrained_model"] = os.path.join(model_dir, prefix)
        inference_config["Global"]["save_inference_dir"] = os.path.join(model_dir, "inference_tmp")
        export(inference_config)
        # Now, export the inference model to ONNX
        model_file = os.path.join(model_dir, "inference_tmp", "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference_tmp", "inference.pdiparams")
        os.makedirs(os.path.join(model_dir, "model/1"), exist_ok=True)
        shutil.copyfile("ppocr/utils/ppocr.pbtxt", os.path.join(model_dir, "model/config.pbtxt")) # TODO: make a generator which will replace the image height and output length using the values from configs
        paddle2onnx.export(model_file, params_file, os.path.join(model_dir, "model/1/model.onnx"))
        triton_log_model(os.path.join(model_dir, "model"), artifact_path="models", await_registration_for=10)
        # Remove local files
        shutil.rmtree(os.path.join(model_dir, "inference_tmp"))
        shutil.rmtree(os.path.join(model_dir, "model"))

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
