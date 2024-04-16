from goldenretriever.data.lit_dataset import GoldenLoader, GoldenStreamingDataset
import transformers as tr
from torch.utils.data import DataLoader

from goldenretriever.data.streaming_dataset import GoldenRetrieverCollator


if __name__ == "__main__":
    dataset = GoldenStreamingDataset(
        name="litdata",
        question_tokenizer=tr.AutoTokenizer.from_pretrained("intfloat/e5-small-v2"),
        input_dir="/home/ric/Projects/golden-retriever/data/dpr-like/el/litdata/train",
        # item_loader=GoldenLoader("intfloat/e5-small-v2"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        collate_fn=GoldenRetrieverCollator(tokenizer=dataset.question_tokenizer),
    )
    
    for sample in dataloader:
        print(sample)
