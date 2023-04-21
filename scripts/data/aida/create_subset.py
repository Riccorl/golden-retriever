from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
docs_sentences = []
with open("/root/golden-retriever/data/aida_dpr_normalized/definitions.txt") as f:
    for idx, line in tqdm(enumerate(f)):
        docs_sentences.append(line)
queries = []
with open("/root/golden-retriever/data/aida_dpr_normalized/definitions_aida.txt") as f:
    for line in tqdm(f):
        queries.append(line)
        # results = retriever.retrieve(query=line, top_k=20, document_store=document_store)
        # for result in results:
        #     set_docs.add(result.content)
        # set_docs.add(line)
set_docs = set()
queries_embeddings = embedder.encode(
    queries, convert_to_tensor=True, show_progress_bar=True, device="cuda"
)
# let's do batches of 100000
for i in range(0, len(docs_sentences), 100000):
    corpus_embeddings = embedder.encode(
        docs_sentences[i : i + 100000],
        convert_to_tensor=True,
        show_progress_bar=True,
        device="cuda",
    )
    # corpus_embeddings = embedder.encode(docs_sentences, convert_to_tensor=True, show_progress_bar=True, device='cuda')

    cosine_scores = util.pytorch_cos_sim(queries_embeddings, corpus_embeddings)
    top_results = torch.topk(cosine_scores, k=50)[1]
    for idx, query in enumerate(queries):
        for doc_idx in top_results[idx, :]:
            set_docs.add(docs_sentences[doc_idx.item()])

# add queries to set
for query in queries:
    set_docs.add(query)
with open(
    "/root/golden-retriever/data/aida_dpr_normalized/definitions_subset.txt", "w"
) as f:
    for doc in set_docs:
        f.write(doc)
