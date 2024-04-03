from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.data.mosaic_datasets import (
    StreamingGoldenRetrieverDataset,
    build_from_hf,
)
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.trainer.train_mosaic import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.data.datasets import AidaInBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder="intfloat/e5-small-v2",
        document_index=InMemoryDocumentIndex(
            documents=DocumentStore.from_file(
                "/home/ric/Projects/golden-retriever/data/dpr-like/el/documents_only_data.jsonl"
            ),
            metadata_fields=["definition"],
            separator=" <def> ",
            device="cuda",
            precision="16",
        ),
    )

    # train_dataset = AidaInBatchNegativesDataset(
    #     name="aida_train",
    #     path="/root/golden-retriever/data/entitylinking/aida_32_tokens_topic/train.jsonl",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     shuffle=True,
    #     use_topics=True,
    # )

    train_dataset = StreamingGoldenRetrieverDataset(
        name="aida_train",
        tokenizer=retriever.question_tokenizer,
        local="data/dpr-like/el/mosaic/train",
        split="train",
        question_batch_size=64,
        # passage_batch_size=400,
    )
    val_dataset = StreamingGoldenRetrieverDataset(
        name="aida_val",
        tokenizer=retriever.question_tokenizer,
        local="data/dpr-like/el/mosaic/val",
        split="train",
        question_batch_size=64,
        # passage_batch_size=400,
    )
    # train_dataset = build_from_hf(
    #     dataset_name="/home/ric/Projects/golden-retriever/data/dpr-like/el/aida_32_tokens_topic/train.jsonl",
    #     split="train",
    #     safe_load=False,
    #     tokenizer=retriever.question_tokenizer,
    #     hf_kwargs={}
    # )
    # val_dataset = build_from_hf(
    #     dataset_name="/home/ric/Projects/golden-retriever/data/dpr-like/el/aida_32_tokens_topic/val.jsonl",
    #     split="train",
    #     safe_load=False,
    #     tokenizer=retriever.question_tokenizer,
    #     hf_kwargs={}
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
        test_dataset=None,
        num_workers=4,
        max_steps=25_000,
        log_to_wandb=False,
        wandb_online_mode=False,
        wandb_project_name="golden-retriever-aida",
        wandb_experiment_name="aida-e5-base-topics-from-blink",
        max_hard_negatives_to_mine=15,
    )

    trainer.train()
    # trainer.test()
