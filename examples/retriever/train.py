from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.trainer import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.data.datasets import InBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(question_encoder="intfloat/e5-small-v2")

    train_dataset = InBatchNegativesDataset(
        name="raco_train",
        path="/root/golden-retriever/data/commonsense/raco/CommonsenseTraining/train.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
    )
    val_dataset = InBatchNegativesDataset(
        name="raco_val",
        path="/root/golden-retriever/data/commonsense/raco/CommonsenseTraining/dev.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
    )

    documents = DocumentStore(ignore_case=True)
    logger.info("Adding documents to document store")
    for sample in tqdm(train_dataset):
        [documents.add_document(s) for s in sample["positives"]]
        [documents.add_document(s) for s in sample["negatives"]]
        [documents.add_document(s) for s in sample["hard_negatives"]]

    for sample in tqdm(val_dataset):
        [documents.add_document(s) for s in sample["positives"]]
        [documents.add_document(s) for s in sample["negatives"]]
        [documents.add_document(s) for s in sample["hard_negatives"]]

    logger.info("Loading document index")
    document_index = InMemoryDocumentIndex(
        documents=documents,
        # metadata_fields=["title"],
        # separator=" <title> ",
        device="cuda",
        precision="16",
    )
    retriever.document_index = document_index

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=0,
        max_steps=25_000,
        wandb_online_mode=True,
        wandb_project_name="golden-retriever-raco",
        wandb_experiment_name="raco-e5-small-inbatch",
        max_hard_negatives_to_mine=0,
    )

    trainer.train()
    trainer.test()
