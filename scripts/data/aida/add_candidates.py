# Path: scripts/data/aida/add_candidates.py
from golden_retriever import GoldenRetriever
import json
import tqdm

if __name__ == "__main__":
    retriever = GoldenRetriever.from_pretrained(
        "retrievers/minilm-l12-fullindex",
        device="cuda:0",
    )
    data_path = "/root/golden-retriever-2/data/dpr-like/el/aida/testb_windowed_3.0.jsonl"
    output_path = "/root/golden-retriever-2/data/dpr-like/el/aida/testb_windowed_candidates.jsonl"
    # with open(output_path, "w") as f:
    #     for line in open(data_path):
    #         sample = json.loads(line)
    #         candidates = retriever.retrieve(sample["doc_text"], k=100)
    #         candidate_titles = [c.split(" <def> ", 1)[0] for c in candidates[0][0]]
    #         sample["candidates"] = candidate_titles
    #         f.write(json.dumps(sample) + "\n")

    # let's do it batched
    retriever.eval()
    batch_size = 32
    documents_batch = []
    with open(output_path, "w") as f:
        for line in tqdm.tqdm(open(data_path)):
            sample = json.loads(line)
            documents_batch.append(sample)
            if len(documents_batch) == batch_size:
                candidates = retriever.retrieve(
                    [d["text"] for d in documents_batch], k=100
                )
                for i, sample in enumerate(documents_batch):
                    candidate_titles = [
                        c.split(" <def>", 1)[0] for c in candidates[0][i]
                    ]
                    sample["window_candidates"] = candidate_titles
                    f.write(json.dumps(sample) + "\n")
                documents_batch = []

        if len(documents_batch) > 0:
            candidates = retriever.retrieve([d["text"] for d in documents_batch], k=100)
            for i, sample in enumerate(documents_batch):
                candidate_titles = [c.split(" <def>", 1)[0] for c in candidates[0][i]]
                sample["window_candidates"] = candidate_titles
                f.write(json.dumps(sample) + "\n")
