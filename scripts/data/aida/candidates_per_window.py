import json


def main(input_path):
    number_of_candidates = 0
    number_of_windows = 0
    with open(input_path, "r") as f:
        for line in f:
            sentence = json.loads(line)
            number_of_windows += 1
            number_of_candidates += len(sentence["positive_ctxs"])

    print(f"Number of candidates: {number_of_candidates}")
    print(f"Number of windows: {number_of_windows}")
    print(
        f"Average number of candidates per window: {number_of_candidates / number_of_windows}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to dpr file")
    args = parser.parse_args()

    # Convert to DPR
    main(args.input)
