from goldenretriever import GoldenRetriever, Trainer
from goldenretriever.common.log import get_logger
from goldenretriever.data.old_datasets import InBatchNegativesDataset

logger = get_logger(__name__)

if __name__ == "__main__":

    retriever = GoldenRetriever(
        question_encoder="intfloat/e5-small-v2", passage_encoder="intfloat/e5-small-v2"
    )

    train_dataset = InBatchNegativesDataset(
        name="webq_train",
        path="/media/hdd1/ric/data/datasets/DPR/downloads/data/retriever/webq-train.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        shuffle=True,
        max_negatives=-1,
        max_hard_negatives=-1,
    )
    val_dataset = InBatchNegativesDataset(
        name="webq_dev",
        path="/media/hdd1/ric/data/datasets/DPR/downloads/data/retriever/webq-dev.json",
        tokenizer=retriever.question_tokenizer,
        question_batch_size=64,
        passage_batch_size=400,
        max_passage_length=64,
        max_negatives=-1,
        max_hard_negatives=-1,
    )

    trainer = Trainer(
        retriever=retriever,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=4,
        max_steps=25_000,
        wandb_online_mode=False,
        wandb_project_name="golden-retriever-dpr",
        wandb_experiment_name="dpr-webq-e5-small",
        max_hard_negatives_to_mine=15,
    )

    trainer.train()
