import json


if __name__ == "__main__":

    with open(
        "/leonardo/home/userexternal/rorland1/golden-retriever-dist/predictions_0.json",
        "r",
    ) as f:
        predictions = json.load(f)

    k = 100
    metrics = {}
    for dataloader_idx, samples in predictions.items():
        hits, total = 0, 0
        for sample in samples:
            # compute the recall at k
            # cut the predictions to the first k elements
            preds = sample["predictions"][:100]
            hits += len(set(preds) & set(sample["gold"]))
            total += len(set(sample["gold"]))

        # compute the mean recall at k
        recall_at_k = hits / total if total > 0 else 0
        metrics[f"recall@{k}_{dataloader_idx}"] = recall_at_k

    metrics[f"recall@{k}"] = (
        sum(metrics.values()) / len(metrics) if len(metrics) > 0 else 0
    )

    print(metrics)
