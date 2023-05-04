from collections import Counter, defaultdict
import json


if __name__ == "__main__":
    path = "experiments/minilm-l12-15hard-400inbatch-42maxlen-tokens-radamw-layernorm-fullindex/2023-04-10/19-30-07/wandb/latest-run/files/predictions/val_debug.json"

    with open("data/dpr-like/el/aida_tokens/train.json") as f:
        train = json.load(f)

    # with open("data/dpr-like/el/aida_tokens/val.json") as f:
    #     val = json.load(f)

    with open(path) as f:
        predictions = json.load(f)

    definitions = []
    for doc in train:
        positives = doc["positive_ctxs"]
        definitions.extend([p["text"] for p in positives])

    totally_missed = []
    for sample in predictions:
        correct_hits = set(sample["gold"]) & set(sample["correct"])
        missed_hits = set(sample["gold"]) - set(sample["correct"])
        totally_missed.extend(missed_hits)

    totally_missed_freq = Counter(totally_missed)

    print("Missed entities:", len(totally_missed_freq))
    print("Missed entities in train:", len(set(totally_missed_freq) & set(definitions)))

    print(
        f"Top frequent wrong prediction: {totally_missed_freq.most_common(1)[0][1]}, {totally_missed_freq.most_common(1)[0][0][:100]} [...]"
    )
    # print the top 10
    print("\nTop 10 wrong predictions:")
    print("freq\tentity")
    for wrong, freq in totally_missed_freq.most_common(10):
        print(f"{freq}\t{wrong[:100]} [...]")

    # count how many times each definition appears in the predictions
    counts = Counter(definitions)

    # wrong_frequncy = []
    # # now we want to know the frequency of each wrong prediction
    # for sample in predictions:
    #     for wrong in sample["wrong"]:
    #         wrong_frequncy.append(wrong)
    # wrong_frequncy = Counter(wrong_frequncy)

    # intersect the two
    wrong_frequency_in_train = defaultdict(int)
    for wrong, freq in totally_missed_freq.items():
        if wrong in counts:
            wrong_frequency_in_train[wrong] = (freq, counts[wrong])
    # sort by frequency in train
    wrong_frequency_in_train = {
        k: v
        for k, v in sorted(
            wrong_frequency_in_train.items(), key=lambda item: item[1][1], reverse=True
        )
    }

    # # print some stats
    # print("Total number of definitions:", len(definitions))
    # print("Total number of unique definitions:", len(set(definitions)))
    # print("Total number of wrong entities:", len(wrong_frequncy))
    # print("Total number of unique wrong entities:", len(set(wrong_frequncy)))
    # print("Total number of wrong entities also in training:", len(wrong_frequency_in_train))
    # print(f"Top frequent entity: {counts.most_common(1)[0][1]}, {counts.most_common(1)[0][0][:100]} [...]")
    # print(f"Top frequent wrong prediction: {wrong_frequncy.most_common(1)[0][1]}, {wrong_frequncy.most_common(1)[0][0][:100]} [...]")
    # # print the top 10
    # print("\nTop 10 wrong predictions:")
    # for wrong, freq in wrong_frequncy.most_common(10):
    #     print(f"{freq}\t{wrong[:100]} [...]")

    print("\nTop 10 wrong predictions that are most frequent in train:")
    # print the top 10
    print("freq\ttrain_freq\tentity")
    for wrong, (freq, train_freq) in list(wrong_frequency_in_train.items())[:10]:
        print(f"{freq}\t{train_freq}\t{wrong[:100]} [...]")

    # are wrong in the train set?
    # print("\nWrong predictions that are also in the train set:")
