import os
from os import path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from ruamel.yaml import YAML

yaml = YAML()

PROJECT_DIR = path.dirname(path.dirname(path.abspath("__FILE__")))


def _load_vector_index(pipeline_config_filepath, embed_model):
    print("START. _load_vector_index()")
    # Load the config
    pipelineconfig = dict()
    with open(pipeline_config_filepath, "r") as file:
        pipelineconfig = yaml.load(file)
    index_folderpath = pipelineconfig.get("filename", "")
    json_folderath = path.join(PROJECT_DIR, "assets", "json", index_folderpath)
    print(f"json_folderath={json_folderath}")

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=json_folderath)
    # load index
    i = load_index_from_storage(storage_context, embed_model=embed_model)
    return i


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


def _load_reranker_model(models_config_filepath):
    print("START. _load_reranker_model()")
    models_config = {}
    with open(models_config_filepath, "r") as f:
        models_config = yaml.load(f)
    rr_config = models_config.get("reranker", {})
    n = rr_config.get("n", 10)
    model_name = rr_config.get("model", "")
    print(f"model_name={model_name}, n={n}")
    # Define
    model_name = rr_config.get("model", "")
    print("_get_reranker_name")

    # Define
    reranker = FlagEmbeddingReranker(
        top_n=n,
        model=model_name,
    )
    print("END. _load_reranker_model()")
    return n, reranker


def _get_query_text(query_filepath):
    query_text = ""
    with open(query_filepath, "r") as f:
        query_text = f.read()
    print("_get_query_text")
    print(f"query={query_text}")
    return query_text


def main():

    # Models
    ### ###
    models_config_filepath = path.join(PROJECT_DIR, "config", "pipeline", "models.yaml")
    # Embedding model
    embed_model = _load_embedding_model(models_config_filepath)
    # LLM
    llm = _load_llm(models_config_filepath)
    Settings.llm = llm
    # Reranker
    topk, reranker = _load_reranker_model(models_config_filepath)

    # Documents / Chunks / Knowledge Base
    ### ###
    # Load Vector Index
    pipeline_config_filepath = path.join(
        PROJECT_DIR, "config", "pipeline", "filename.yaml"
    )
    docs_vector_index = _load_vector_index(pipeline_config_filepath, embed_model)

    # Query engine & Query
    ### ###
    recursive_query_engine = docs_vector_index.as_query_engine(
        similarity_top_k=topk,
        node_postprocessors=[reranker],
        verbose=True,
    )
    # Generally, if the prompt text is specified, the quality of the parsed document is better.
    query_filepath = path.join(
        PROJECT_DIR,
        "assets",
        "prompts",
        "query.txt",
    )
    query = _get_query_text(query_filepath)

    # Response
    ### ###
    response = recursive_query_engine.query(query)
    print(response)

    print("Done")


if __name__ == "__main__":
    main()
