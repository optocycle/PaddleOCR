[<img src="https://img.shields.io/badge/Language-English-blue.svg">](README_en.md) | [<img src="https://img.shields.io/badge/Language-简体中文-red.svg">](README.md)

<p align="center">
 <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/PaddleOCR_logo.png" align="middle" width = "600"/>
<p align="center">
<p align="center">
    <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>

# Optocycle PaddleOCR Fork

This fork includes configurations for fine-tuning PaddleOCR on license plate data, along with an extension for MLflow logging.

## Fine-tuning a model

These instructions are based on the guide provided [here](https://paddlepaddle.github.io/PaddleOCR/main/en/ppocr/model_train/recognition.html). 
1. Open the cloned repository and install the environment with `poetry`: 
```bash
cd PaddleOCR/
poetry install
```
2. Download a pretrained model.
```bash
# Download the pre-trained model for en_PP-OCRv3 (the weights can also be found in the home bucket on MinIO)
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar
# Decompress model parameters
cd pretrain_models
tar -xf en_PP-OCRv3_rec_train.tar && rm -rf en_PP-OCRv3_rec_train.tar
cd ..
```
3. Ensure your dataset is formatted for PaddleOCR and move it to the `datasets/` directory: `mv /path/to/paddleocr_dataset ./datasets`
4. Fine-tune the model:
```bash
poetry run python3 tools/train.py -c configs/rec/PP-OCRv3/slplates_finetuning.yml -o Global.pretrained_model=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy
```
5. Export the fine-tuned model to an inference model and specify a custom destination directory (e.g., `./slplates_inference_model`):
```bash
poetry run python3 tools/export_model.py -c configs/rec/PP-OCRv3/slplates_finetuning.yml -o Global.pretrained_model=output/normal_finetuned_slplates_paddleocr/best_model/model  Global.save_inference_dir=./slplates_inference_model
```

## Enabling MLflow

To enable MLflow logging during fine-tuning, add the following to your configuration file:
```yaml
mlflow:
  mlflow_log_every_n_iter: <n_iter>  # Log training metrics every <n_iter> iterations (evaluation metrics are always logged)
  name: <name_of_the_run>  # Name of the MLflow run (e.g., "lp_finetuned")
  log_only_best_model: <true/false>  # Whether to log only the best model based on its accuracy
  pbtxt_configs:  # Configuration for generating .pbtxt files to log models in Triton inference server format
    max_batch_size: <n_samples>  # Maximum batch size for inference
    dynamic_batching: <true/false>  # Enable or disable dynamic batching for Triton
```
Additionally, create a `.env` file by copying `.env.example` and filling in the correct values."

## 📄 License

This project is released under [Apache License Version 2.0](./LICENSE).

### Changes
In addition to the modifications described in the header of the original files, the following changes have been made:
- The `setup.py` and `MANIFEST.in` files have been removed as the project has been switched to using Poetry for dependency management and packaging.
- Heavily revised `README.md` to reflect changes and usage for license plate fine-tuning.