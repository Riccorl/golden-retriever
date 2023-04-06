import json
import random


if __name__ == "__main__":
    with open("data/aida_dpr_special_tok/definitions.txt") as f:
        definitions = [line.strip() for line in f]

    with open("data/aida_dpr_special_tok/train.json") as f:
        train = json.load(f)

    with open("data/aida_dpr_special_tok/val.json") as f:
        val = json.load(f)

    with open("data/aida_dpr_special_tok/test.json") as f:
        test = json.load(f)

    data_definitons = set()
    for data in [train, val, test]:
        for sample in data:
            for positive in sample["positive_ctxs"]:
                data_definitons.add(positive["text"].strip())

    random_def = random.sample(definitions, 1_000_000)

    # merge the two
    final_definitions = list(data_definitons) + random_def
    # remove duplicates
    final_definitions = list(set(final_definitions))

    with open("data/aida_dpr_special_tok/subsample_definitions.txt", "w") as f:
        f.writelines(f"{def_}\n" for def_ in final_definitions)
