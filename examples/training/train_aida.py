from tqdm import tqdm
from goldenretriever.common.log import get_logger
# from goldenretriever.data.streaming_dataset import StreamingGoldenRetrieverDataset
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.trainer import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.data.datasets import AidaInBatchNegativesDataset
import os

from lightning.pytorch.strategies import FSDPStrategy

logger = get_logger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder="intfloat/e5-small-v2",
        use_hf_model=True,
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

    strategy = FSDPStrategy(
        # # Default: Shard weights, gradients, optimizer state (1 + 2 + 3)
        # sharding_strategy="FULL_SHARD",
        # # Shard gradients, optimizer state (2 + 3)
        # sharding_strategy="SHARD_GRAD_OP",
        # # Full-shard within a machine, replicate across machines
        # sharding_strategy="HYBRID_SHARD",
        # Don't shard anything (similar to DDP)
        sharding_strategy="NO_SHARD",
    )

    trainer = Trainer(
        retriever=retriever,
        train_dataset=None,
        val_dataset=None,
        # test_dataset=test_dataset,
        num_workers=8,
        max_steps=25_000,
        # log_to_wandb=False,
        wandb_online_mode=False,
        wandb_project_name="golden-retriever-aida",
        wandb_experiment_name="aida-e5-base-topics-from-blink",
        max_hard_negatives_to_mine=0,
        # strategy="deepspeed_stage_2",
        strategy=strategy,
        devices=2,
        # strategy="ddp_find_unused_parameters_true",
        # devices=2,
        # accelerator="cuda",
    )

    trainer.train()

    # train_dataset = AidaInBatchNegativesDataset(
    #     name="aida_train",
    #     path="/home/ric/Projects/golden-retriever/data/dpr-like/el/aida_32_tokens_topic/train.jsonl",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     shuffle=True,
    #     use_topics=True,
    #     prefetch=False,
    # )
    # val_dataset = AidaInBatchNegativesDataset(
    #     name="aida_val",
    #     path="/home/ric/Projects/golden-retriever/data/dpr-like/el/aida_32_tokens_topic/val.jsonl",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     use_topics=True,
    #     # prefetch=False,
    # )
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

    # train_dataset = StreamingGoldenRetrieverDataset(
    #     name="aida_train",
    #     tokenizer=retriever.question_tokenizer,
    #     local="data/dpr-like/el/mosaic/train",
    #     split="train",
    #     question_batch_size=32,
    #     passage_batch_size=400,
    # )
    # val_dataset = StreamingGoldenRetrieverDataset(
    #     name="aida_val",
    #     tokenizer=retriever.question_tokenizer,
    #     local="data/dpr-like/el/mosaic/val",
    #     split="train",
    #     question_batch_size=32,
    #     passage_batch_size=400,
    # )
    
    
    # trainer.test()
