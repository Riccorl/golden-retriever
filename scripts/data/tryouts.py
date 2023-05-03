import transformers

from golden_retriever import GoldenRetriever

# tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
# model = transformers.BertModel.from_pretrained("bert-base-uncased")

model = GoldenRetriever.from_pretrained(
    "/media/hdd1/ric/models/golden-retriever/retrievers/mpnet-24words-topics-rebel-1encoder",
    device="cuda",
    index_device="cuda",
    index_precision="fp16",
)

model.retrieve("This is just a test mr Obama!")
