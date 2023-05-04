import json
import argparse


def add_topics(input_path, entqa_path, output_path):
    # with open(input_path) as f:
    #     input_data = json.load(f)

    input_data = []
    with open(input_path) as f:
        for line in f:
            input_data.append(json.loads(line))

    entqa_data = []
    with open(entqa_path) as f:
        for line in f:
            entqa_data.append(json.loads(line))

    topics = {}
    for sample in entqa_data:
        topics[f"{sample['doc_id']}_{sample['window_id']}"] = sample["doc_topic"]

    for sample in input_data:
        sample["doc_topic"] = topics[f"{sample['doc_id']}_{sample['window_id']}"]

    # dump json line
    with open(output_path, "w") as f:
        # json.dump(input_data, f)
        for sample in input_data:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str, required=True)
    arg_parser.add_argument("--entqa_path", type=str, required=True)
    arg_parser.add_argument("--output_path", type=str, required=True)

    add_topics(**vars(arg_parser.parse_args()))
