from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.trainer import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.data.datasets import (
    InBatchNegativesDataset,
    AidaInBatchNegativesDataset,
    SubsampleStrategyEnum,
)

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(question_encoder="intfloat/e5-base-v2")

    train_dataset = AidaInBatchNegativesDataset(
        name="aida_train",
        # path="/media/data/EL/blink/window_32_tokens/random_1M/dpr-like/first_1M.jsonl",
        path="/media/data/EL/blink/window_32_tokens_25032024/train.fixed.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=32,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
        subsample_strategy=SubsampleStrategyEnum.IN_ORDER,
        subsample_portion=100_000,
        prefetch=True
        # use_topics=True,
    )
    val_dataset = AidaInBatchNegativesDataset(
        name="aida_val",
        path="/media/data/EL/blink/window_32_tokens/random_1M/dpr-like/val.jsonl",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        # use_topics=True,
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

    logger.info("Loading document index")
    document_index = InMemoryDocumentIndex(
        documents=DocumentStore.from_file(
            "/root/golden-retriever/data/entitylinking/documents.jsonl"
        ),
        metadata_fields=["definition"],
        separator=" <def> ",
        device="cuda",
        precision="16",
    )
    retriever.document_index = document_index

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        num_workers=0,
        max_steps=600_000,
        early_stopping=False,
        wandb_online_mode=True,
        wandb_project_name="golden-retriever-blink",
        wandb_experiment_name="blink-coverage-e5-base-topics",
        max_hard_negatives_to_mine=15,
        mine_hard_negatives_with_probability=0.2,
        accumulate_grad_batches=2,
        save_last=True,
        # resume_from_checkpoint_path="/root/golden-retriever/wandb/run-20240326_105401-34vldlhh/files/checkpoints/checkpoint-validate_recall@100_0.9542-epoch_11.ckpt",
    )

    trainer.train()
    trainer.test()
