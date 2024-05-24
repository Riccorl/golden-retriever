from tqdm import tqdm
from goldenretriever.common.log import get_logger
from goldenretriever.indexers.base import BaseDocumentIndex
from goldenretriever.indexers.document import DocumentStore
from goldenretriever.trainer.train import Trainer
from goldenretriever import GoldenRetriever
from goldenretriever.indexers.inmemory import InMemoryDocumentIndex
from goldenretriever.indexers.faiss import FaissDocumentIndex
from goldenretriever.data.datasets import InBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":
    # instantiate retriever
    # document_index = BaseDocumentIndex.from_pretrained(
    #     "wandb/run-20240521_212032-fqnhon6y/files/retriever/document_index",
    # )
    retriever = GoldenRetriever(
        question_encoder="wandb/run-20240521_212032-fqnhon6y/files/retriever/question_encoder",
        document_index="wandb/run-20240521_212032-fqnhon6y/files/retriever/document_index",
        device="cuda",
        precision="16",
    )

    # train_dataset = InBatchNegativesDataset(
    #     name="raco_train",
    #     path="/root/golden-retriever/data/commonsense/raco/CommonsenseTraining/train.json",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    #     shuffle=True,
    # )
    # val_dataset = InBatchNegativesDataset(
    #     name="raco_val",
    #     path="/root/golden-retriever/data/commonsense/raco/CommonsenseTraining/dev.json",
    #     tokenizer=retriever.question_tokenizer,
    #     question_batch_size=64,
    #     passage_batch_size=400,
    #     max_passage_length=64,
    # )
    test_dataset = InBatchNegativesDataset(
        name="raco_val",
        path="/media/data/commonsense/RACo/data/dev.json",
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
        train_dataset=None,
        val_dataset=None,
        test_dataset=test_dataset,
        num_workers=0,
        max_steps=25_000,
        wandb_online_mode=False,
        wandb_project_name="golden-retriever-raco",
        wandb_experiment_name="raco-e5-base-eval",
        max_hard_negatives_to_mine=0,
        top_k=[1, 3, 5, 10, 20, 50, 100]
    )

    # trainer.train()
    trainer.test()
