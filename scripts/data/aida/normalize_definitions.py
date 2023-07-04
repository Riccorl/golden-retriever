import json

from tqdm import tqdm


if __name__ == "__main__":
    definitions_dpr = set()
    for set in ["train", "val", "test"]:
        with open(f"/root/golden-retriever/data/aida_dpr/{set}.json", "r") as f:
            data = json.load(f)

        for sample in data:
            for positive_pssg in sample["positive_pssgs"]:
                positive_pssg["text"] = positive_pssg["text"].strip()

        with open(
            f"/root/golden-retriever/data/aida_dpr_normalized/{set}.json", "w"
        ) as f:
            json.dump(data, f, indent=2)

    with open("/root/golden-retriever/data/aida_dpr/definitions_aida.txt") as f:
        definitions = f.readlines()

    with open(
        "/root/golden-retriever/data/aida_dpr_normalized/definitions_aida.txt", "w"
    ) as f:
        for definition in definitions:
            definition = definition.strip()
            f.write(definition + "\n")

    with open("/root/golden-retriever/data/aida_dpr/definitions.txt") as f:
        definitions = f.readlines()

    with open(
        "/root/golden-retriever/data/aida_dpr_normalized/definitions.txt", "w"
    ) as f:
        for definition in tqdm(definitions):
            definition = definition.strip()
            f.write(definition + "\n")
