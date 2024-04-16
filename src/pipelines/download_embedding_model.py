"""
    Pre-requisites to download Gemma 2B
    1. You have a huggingface account
    2. You have an auth token from HF
    3. You have huggingface-cli installed. Try brew install huggingface-cli
    
    Before you start:
    1. Login to HF Hub via huggingface-cli login
"""

from os import path
from sentence_transformers import SentenceTransformer

from ruamel.yaml import YAML

yaml = YAML()
PROJECT_DIR = path.dirname(path.dirname(path.abspath("__FILE__")))


def _get_model_name(models_config_filepth):
    models_config = {}
    with open(models_config_filepth, "r") as f:
        models_config = yaml.load(f)
    model_name = models_config.get("embed", {}).get("model", "")
    print("_get_model_name")
    print(f"model_name={model_name}")
    return model_name


def main():
    # Get filename
    models_config_filepth = path.join(PROJECT_DIR, "config", "pipeline", "models.yaml")
    model_name = _get_model_name(models_config_filepth)
    # Define
    MODELS_FOLDERPATH = path.join(PROJECT_DIR, "assets", "llm")
    save_path = path.join(MODELS_FOLDERPATH, model_name)
    print(f"save_path={save_path}")

    # Download
    model = SentenceTransformer(model_name)
    print("Loaded model")

    # Save
    model.save(save_path)
    print(f"Model saved at {save_path}")
    print("Done")


if __name__ == "__main__":
    main()
