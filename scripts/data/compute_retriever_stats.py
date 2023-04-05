import json
import collections


def compute_retriever_stats(dataset_path: str) -> None:
    correct, total = 0, 0
    with open(dataset_path) as f:
        for line in f:
            window_info = json.loads(line.strip())
            window_candidates = window_info["window_candidates"]
            window_candidates = [c.replace(" ", "_").lower() for c in window_candidates]

            for ss, se, label in window_info["window_labels"]:
                if label == "--NME--":
                    continue
                if label.lower() in window_candidates:
                    correct += 1
                total += 1

    recall = correct / total
    print("Recall:", recall)


def main():
    import sys

    dataset_path = sys.argv[1]
    compute_retriever_stats(dataset_path)


if __name__ == "__main__":
    main()
