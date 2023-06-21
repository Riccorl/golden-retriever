import argparse
import json
from pathlib import Path
from tqdm import tqdm


def check_definitions(
    data_path, definitions_path, cleaned_data_path, cleaned_definitions_path
):
    with open(data_path, "r") as f:
        data = json.load(f)

    with open(definitions_path, "r") as f:
        definitions = [line.strip() for line in f]

    definitions = set(definitions)

    missing_definitions = []
    for sample in tqdm(data):
        for defs in sample["positive_ctxs"]:
            text = defs["text"]
            # clean text
            text = text.strip()
            entity, entity_def = text.split("<def>")
            # clean
            entity = entity.strip()
            entity_def = entity_def.strip()
            # merge again
            text = f"{entity} <def> {entity_def}"
            # clean again
            text = text.strip()
            # check if in definitions
            if text not in definitions:
                missing_definitions.append(text)
                # also add to definitions
                definitions.add(text)
            # update text
            defs["text"] = text

    # write definitions
    cleaned_definitions_path = Path(cleaned_definitions_path)
    cleaned_definitions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cleaned_definitions_path, "w") as f:
        for definition in definitions:
            f.write(f"{definition}\n")

    # write data
    cleaned_data_path = Path(cleaned_data_path)
    cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cleaned_data_path, "w") as f:
        for sample in data:
            f.write(f"{json.dumps(sample)}\n")

    print("Missing definitions:")
    for definition in missing_definitions:
        print(definition)
    print("Number of missing definitions:", len(missing_definitions))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", type=str, required=True)
    arg_parser.add_argument("--definitions_path", type=str, required=True)
    arg_parser.add_argument("--cleaned_data_path", type=str, required=True)
    arg_parser.add_argument("--cleaned_definitions_path", type=str, required=True)

    check_definitions(**vars(arg_parser.parse_args()))
