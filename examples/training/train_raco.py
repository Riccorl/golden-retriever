from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.indexers.document import DocumentStore
from goldenretriever import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.data.datasets import AidaInBatchNegativesDataset, InBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(
        # question_encoder="facebook/contriever",
        question_encoder="intfloat/e5-base-v2",
        document_index=InMemoryDocumentIndex(
            documents=DocumentStore.from_file(
                "/media/data/commonsense/RACo/data/train.dev.index.jsonl"
            ),
            # metadata_fields=["definition"],
            # separator=" <def> ",
            device="cuda",
            precision="16",
        ),
    )

    train_dataset = InBatchNegativesDataset(
        name="raco_train",
        path="/media/data/commonsense/RACo/data/train.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=200,
        max_passage_length=64,
        shuffle=True,
        # load_from_cache_file=False,
    )
    val_dataset = InBatchNegativesDataset(
        name="raco_val",
        path="/media/data/commonsense/RACo/data/dev.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=200,
        max_passage_length=64,
        # load_from_cache_file=False,
    )
    # test_dataset = AidaInBatchNegativesDataset(
    #     name="aida_test",
    #     path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/test.jsonl",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     use_topics=True,
    # )

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
        num_workers=0,
        max_steps=55_000,
        wandb_online_mode=True,
        wandb_log_model=False,
        wandb_project_name="golden-retriever-raco",
        # wandb_experiment_name="e5-base-v2-raco-self-index-hn=5-prob=0.2-batch=400",
        wandb_experiment_name="e5-base-v2-raco-self-index-batch=400",
        max_hard_negatives_to_mine=0,
        # mine_hard_negatives_with_probability=0.2,
    )

    trainer.train()
