from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.indexers.document import DocumentStore
from goldenretriever import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.data.datasets import AidaInBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder="riccorl/e5-base-v2-blink-1M-32words-windows",
        document_index=InMemoryDocumentIndex(
            documents=DocumentStore.from_file(
                "/root/golden-retriever/data/entitylinking/documents.jsonl"
            ),
            metadata_fields=["definition"],
            separator=" <def> ",
            device="cuda",
            precision="16",
        ),
    )

    train_dataset = AidaInBatchNegativesDataset(
        name="aida_train",
        path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/train.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
        use_topics=True,
    )
    val_dataset = AidaInBatchNegativesDataset(
        name="aida_val",
        path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/val.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=True,
    )
    test_dataset = AidaInBatchNegativesDataset(
        name="aida_test",
        path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/test.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        use_topics=True,
    )

    # logger.info("Loading document index")
    # document_index = InMemoryDocumentIndex(
    #     documents=DocumentStore.from_file("/root/golden-retriever/data/entitylinking/documents.jsonl"),
    #     metadata_fields=["definition"],
    #     separator=" <def> ",
    #     device="cuda",
    #     precision="16",
    # )
    # retriever.document_index = document_index

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        max_steps=25_000,
        wandb_online_mode=True,
        wandb_log_model=False,
        wandb_project_name="golden-retriever-aida",
        wandb_experiment_name="aida-e5-base-topics-from-blink-1M-32words-windows",
        max_hard_negatives_to_mine=15,
    )

    trainer.train()
    trainer.test()
