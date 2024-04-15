from goldenretriever.data.lit_dataset import GoldenLoader, GoldenStreamingDataset
import transformers as tr

if __name__ == "__main__":
    dataset = GoldenStreamingDataset(
        name="litdata",
        question_tokenizer=tr.AutoTokenizer.from_pretrained("intfloat/e5-small-v2"),
        input_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata",
        # item_loader=GoldenLoader("intfloat/e5-small-v2"),
    )
    
    for sample in dataset:
        print(sample)
