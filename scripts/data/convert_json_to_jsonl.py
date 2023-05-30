import argparse
import json

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file_path", type=str)
    arg_parser.add_argument("output_file_path", type=str)
    args = arg_parser.parse_args()

    # convert json to jsonl
    print("Reading data from", args.input_file_path)
    with open(args.input_file_path) as f:
        data = json.load(f)
    
    print("Writing data to", args.output_file_path)
    with open(args.output_file_path, "w") as f:
        for doc in data:
            f.write(json.dumps(doc) + "\n")
            