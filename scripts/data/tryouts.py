import transformers

from golden_retriever import GoldenRetriever

# tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
# model = transformers.BertModel.from_pretrained("bert-base-uncased")

model = GoldenRetriever.from_pretrained("riccorl/golden-retriever-mpnet-aida-24words-topics-crossencoder")
