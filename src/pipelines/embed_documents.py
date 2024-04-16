import pickle
import json
import os
from os import path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser, SentenceSplitter

from ruamel.yaml import YAML

yaml = YAML()

PROJECT_DIR = path.dirname(path.dirname(path.abspath("__FILE__")))


def _load_parsed_document(pipeline_config_filepath):
    print("START. _load_parsed_document()")
    # Load the config
    pipelineconfig = dict()
    with open(pipeline_config_filepath, "r") as file:
        pipelineconfig = yaml.load(file)
    document_filename = pipelineconfig.get("filename", "")
    document_filename = str(document_filename)
    print(f"document_filename={document_filename}")

    document_pkl_filepath = path.join(
        PROJECT_DIR,
        "assets",
        "pickle",
        f"{document_filename}.pkl",
    )
    print(f"pkl_path={document_pkl_filepath}")
    parser_output = None
    if os.path.exists(document_pkl_filepath):
        print("path exists. load pkl file.")
        with open(document_pkl_filepath, "rb") as file:
            parser_output = pickle.load(file)
        print("file loaded. END.")
        return parser_output
    else:
        raise FileNotFoundError("path does not exist.")


def _save_document_nodes(
    vector_store_index: VectorStoreIndex, pipeline_config_filepath
):
    print("START. _save_document_nodes()")
    # Load the config
    pipelineconfig = dict()
    with open(pipeline_config_filepath, "r") as file:
        pipelineconfig = yaml.load(file)
    document_filename = pipelineconfig.get("filename", "")
    document_filename = str(document_filename)
    document_filename = document_filename
    print(f"document_filename={document_filename}")

    json_folderath = path.join(PROJECT_DIR, "assets", "json", document_filename)
    print(f"json_folderath={json_folderath}")
    vector_store_index.storage_context.persist(json_folderath)
    print("file saved. END.")
    return True


def _load_embedding_model(models_config_filepath):
    print("START. _load_embedding_model()")
    models_config = {}
    with open(models_config_filepath, "r") as f:
        models_config = yaml.load(f)
    model_name = models_config.get("embed", {}).get("model", "")
    print("_get_model_name")
    print(f"model_name={model_name}")
    # Define
    MODELS_FOLDERPATH = path.join(PROJECT_DIR, "assets", "llm")
    model_path = path.join(MODELS_FOLDERPATH, model_name)
    model = HuggingFaceEmbedding(model_name=model_path)
    print("model loaded. END.")
    return model


def _load_llm(models_config_filepath):
    print("START. _load_llm()")
    models_config = {}
    with open(models_config_filepath, "r") as f:
        models_config = yaml.load(f)
    llm_config = models_config.get("llm", {})
    provider = llm_config.get("c", "")
    model_name = llm_config.get("model", "")
    print(f"provider={provider}, model={model_name}")
    # Anthropic (Claude)
    if provider == "anthropic":
        anthropic_secrets_path = path.join(
            PROJECT_DIR, "config", "auth", "anthropic.yaml"
        )
        anthropic_auth_config = {}
        with open(anthropic_secrets_path) as f:
            anthropic_auth_config = yaml.load(f)
        claude_api_key = anthropic_auth_config.get("key", "")
        os.environ["ANTHROPIC_API_KEY"] = claude_api_key
        print("model loaded. END.")
        return Anthropic(model=model_name)


def main():

    # Parsed document
    pipeline_config_filepath = path.join(
        PROJECT_DIR, "config", "pipeline", "filename.yaml"
    )
    parsed_document = _load_parsed_document(
        pipeline_config_filepath=pipeline_config_filepath
    )

    # Models (Embedding, LLM)
    models_config_filepath = path.join(PROJECT_DIR, "config", "pipeline", "models.yaml")
    # Embedding model
    embed_model = _load_embedding_model(models_config_filepath)
    Settings.embed_model = embed_model
    # LLM model
    llm = _load_llm(models_config_filepath)
    Settings.llm = llm

    # Node parser

    # Baseline
    # --------
    # node_parser = MarkdownElementNodeParser(llm=llm, num_workers=1)
    # nodes = node_parser.get_nodes_from_documents(parsed_document, progress=True)
    # base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    # recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
    # --------
    # End of Baseline

    # Experiment 1
    # --------
    node_parser = SentenceSplitter(
        chunk_size=64,  # minimum: 50
        chunk_overlap=0,
    )
    nodes = node_parser.get_nodes_from_documents(parsed_document)
    recursive_index = VectorStoreIndex(nodes=nodes)
    # --------
    # End of Experiment 1

    # Save
    if _save_document_nodes(
        recursive_index, pipeline_config_filepath=pipeline_config_filepath
    ):
        print("Done")


if __name__ == "__main__":
    main()
