import pickle
from os import path
from llama_parse import LlamaParse

from ruamel.yaml import YAML

yaml = YAML()

PROJECT_DIR = path.dirname(path.dirname(path.abspath("__FILE__")))


def _get_llamaindex_api_key(config_filepath):
    config = dict()
    with open(config_filepath, "r") as file:
        config = yaml.load(file)
    api_key = config.get("key", "")
    return api_key


def _get_prompt_text(prompt_filepath):
    with open(prompt_filepath, "r") as file:
        instruction_text = file.read()
    print("_get_prompt_text")
    print(f"instructions={instruction_text}")
    return instruction_text


def _get_document_filename(pipeline_config_filepath):
    pipelineconfig = dict()
    with open(pipeline_config_filepath, "r") as file:
        pipelineconfig = yaml.load(file)
    document_filename = pipelineconfig.get("filename", "")
    print("_get_document_filename")
    print(f"document_filename={document_filename}")
    return document_filename


def _parse_document_or_load(api_key, pkl_path, document_path, instruction_text):
    print(f"pkl_path={pkl_path}")
    parser_output = None
    if not path.exists(pkl_path):
        print("path does not exist. run parsing in llamacloud.")
        # Perform the parsing step and store the result in llama_parse_documents
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction=instruction_text,
        )
        print(f"document_path={document_path}")
        parser_output = parser.load_data(document_path)
        with open(pkl_path, "wb") as file:
            pickle.dump(parser_output, file)
        print("saved.")
    else:

        print("path exists. load pkl file.")
        with open(pkl_path, "rb") as file:
            parser_output = pickle.load(file)
        print("loaded.")
    return parser_output


def main():

    # Get API key
    llamaconfig_filepath = path.join(PROJECT_DIR, "config", "auth", "llamaindex.yaml")
    api_key = _get_llamaindex_api_key(llamaconfig_filepath)

    # Get prompt text.
    # Generally, if the prompt text is specified, the quality of the parsed document is better.
    prompt_filepath = path.join(
        PROJECT_DIR,
        "assets",
        "prompts",
        "extract.txt",
    )
    instruction_text = _get_prompt_text(prompt_filepath)

    # Get filename
    pipeline_config_filepath = path.join(
        PROJECT_DIR, "config", "pipeline", "filename.yaml"
    )
    document_filename = _get_document_filename(pipeline_config_filepath)
    # Parse document
    document_path = path.join(
        PROJECT_DIR,
        "assets",
        "inventory",
        f"{document_filename}.pdf",
    )
    pkl_path = path.join(
        PROJECT_DIR,
        "assets",
        "pickle",
        f"{document_filename}.pkl",
    )

    parsed_document = _parse_document_or_load(
        api_key=api_key,
        pkl_path=pkl_path,
        document_path=document_path,
        instruction_text=instruction_text,
    )
    first_document = parsed_document[0].text
    print(len(first_document))
    if 1000 < len(first_document):
        print(first_document[:1000])
    else:
        print(first_document)

    print("Done")


if __name__ == "__main__":
    main()
