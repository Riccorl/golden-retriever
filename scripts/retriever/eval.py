from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.trainer import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.indexers.faiss import FaissDocumentIndex
from goldenretriever.data.datasets import InBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    document_index = BaseDocumentIndex.from_pretrained(
        "/root/golden-retriever/data/retrievers/raco-e5-small-v2/training-index/document_index",
        # _target_="goldenretriever.indexers.faiss.FaissDocumentIndex"
    )
    retriever = GoldenRetriever(
        question_encoder="/root/golden-retriever/data/retrievers/raco-e5-small-v2/training-index/question_encoder",
        document_index=document_index,
        device="cuda",
        precision="16",
    )

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
    test_dataset = InBatchNegativesDataset(
        name="raco_val",
        path="/root/golden-retriever/data/commonsense/raco/CommonsenseTraining/obqa.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
    )

    # logger.info("Loading document index")
    # document_index = InMemoryDocumentIndex(
    #     documents=documents,
    #     # metadata_fields=["title"],
    #     # separator=" <title> ",
    #     device="cuda",
    #     precision="16",
    # )
    # retriever.document_index = document_index

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_workers=0,
        max_steps=25_000,
        wandb_online_mode=False,
        wandb_project_name="golden-retriever-raco",
        wandb_experiment_name="raco-e5-small-inbatch",
        max_hard_negatives_to_mine=0,
        top_ks=[5, 10]
    )

    # trainer.train()
    trainer.test()
