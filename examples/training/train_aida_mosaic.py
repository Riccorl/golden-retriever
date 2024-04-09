from composer.utils import dist, get_device
from tqdm import tqdm

from goldenretriever import GoldenRetriever
from goldenretriever.common.log import get_logger
from goldenretriever.data.old_datasets import AidaInBatchNegativesDataset
from goldenretriever.data.datasets import (
    GoldenRetrieverStreamingDataset,
)
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.indexers.faiss_index import FaissDocumentIndex
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.trainer.train_mosaic import Trainer

logger = get_logger(__name__)

if __name__ == "__main__":
    # dist.initialize_dist(get_device(None))  # , timeout=600)
    # instantiate retriever
    retriever = GoldenRetriever(
        question_encoder="intfloat/e5-small-v2",
        document_index=InMemoryDocumentIndex(
            documents=DocumentStore.from_file("data/el/documents.jsonl"),
            metadata_fields=["definition"],
            separator=" <def> ",
            device="cuda:1",
            precision="16",
        ),
        # document_index=FaissDocumentIndex(
        #     documents=DocumentStore.from_file(
        #         "data/el/documents_only_data.jsonl"
        #     ),
        #     metadata_fields=["definition"],
        #     separator=" <def> ",
        # )
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

    # train_dataset = GoldenRetrieverStreamingDataset(
    #     name="aida_train",
    #     tokenizer=retriever.question_tokenizer,
    #     local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/train",
    #     split="train",
    #     batch_size=32,
    #     shuffle=True,
    #     shuffle_seed=42,
    # )
    # val_dataset = GoldenRetrieverStreamingDataset(
    #     name="aida_val",
    #     tokenizer=retriever.question_tokenizer,
    #     local="/home/ric/Projects/golden-retriever/data/dpr-like/el/mosaic/val",
    #     split="train",
    #     batch_size=32,
    # )
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
        train_dataset="data/el/mosaic/train",
        train_dataset_kwargs={"predownload": 10_000},
        train_batch_size=128,
        device_train_microbatch_size=64,
        val_dataset="data/el/mosaic/val",
        val_batch_size=128,
        test_dataset=None,
        num_workers=8,
        max_duration="25000ep",
        eval_interval="1ep",
        # eval_interval="10ba",
        log_to_wandb=True,
        wandb_online_mode=False,
        wandb_project_name="golden-retriever-aida",
        wandb_experiment_name="aida-e5-base-topics-from-blink",
        max_hard_negatives_to_mine=15,
        save_top_k=-1,
        # deepspeed_config={
        #     "train_batch_size": 64,
        #     "train_micro_batch_size_per_gpu": 32,
        #     "gradient_accumulation_steps": 1,
        #     "fp16": {"enabled": True},
        #     "zero_optimization": {
        #         "stage": 1,
        #         # "offload_optimizer": {"device": "cpu", "pin_memory": True},
        #     },
        # },
    )

    trainer.train()
    # trainer.test()
