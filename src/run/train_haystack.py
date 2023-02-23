import argparse
import logging
import os
from typing import Union

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)


def train(
    doc_dir: Union[str, os.PathLike],
    train_filename: Union[str, os.PathLike],
    dev_filename: Union[str, os.PathLike],
    save_dir: Union[str, os.PathLike],
    query_model: str = "facebook/dpr-question_encoder-single-nq-base",
    passage_model: str = "facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query: int = 256,
    max_seq_len_passage: int = 256,
    max_epochs: int = 1,
    batch_size: int = 16,
    grad_acc_steps: int = 8,
    evaluate_every: int = 3000,
    embed_title: bool = True,
    num_positives: int = 1,
    num_hard_negatives: int = 0,
):
    # Initialize DPR model
    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(),
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=max_seq_len_query,
        max_seq_len_passage=max_seq_len_passage,
        batch_size=batch_size,
    )

    # Start training our model and save it when it is finished
    retriever.train(
        data_dir=doc_dir,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=dev_filename,
        n_epochs=max_epochs,
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        save_dir=save_dir,
        evaluate_every=evaluate_every,
        embed_title=embed_title,
        num_positives=num_positives,
        num_hard_negatives=num_hard_negatives,
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--doc_dir",
        type=str,
        help="Path to directory containing documents in json format",
    )
    arg_parser.add_argument(
        "--train_filename",
        type=str,
        default="train.json",
        help="Filename of training data",
    )
    arg_parser.add_argument(
        "--dev_filename",
        type=str,
        default="dev.json",
        help="Filename of development data",
    )
    arg_parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments/hystack",
        help="Path to directory where model should be saved",
    )
    arg_parser.add_argument(
        "--query_model",
        type=str,
        default="facebook/dpr-question_encoder-single-nq-base",
        help="Name of query model",
    )
    arg_parser.add_argument(
        "--passage_model",
        type=str,
        default="facebook/dpr-ctx_encoder-single-nq-base",
        help="Name of passage model",
    )
    arg_parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        help="Number of epochs to train for",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    arg_parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    arg_parser.add_argument(
        "--evaluate_every",
        type=int,
        default=3000,
        help="Evaluate model every n steps",
    )
    arg_parser.add_argument(
        "--embed_title",
        default="store_true",
        help="Whether to embed title in passage",
    )
    arg_parser.add_argument(
        "--num_positives",
        type=int,
        default=1,
        help="Number of positive passages per query",
    )
    arg_parser.add_argument(
        "--num_hard_negatives",
        type=int,
        default=0,
        help="Number of hard negative passages per query",
    )
    args = arg_parser.parse_args()

    train(
        doc_dir=args.doc_dir,
        train_filename=args.train_filename,
        dev_filename=args.dev_filename,
        save_dir=args.save_dir,
        query_model=args.query_model,
        passage_model=args.passage_model,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        evaluate_every=args.evaluate_every,
        embed_title=args.embed_title,
        num_positives=args.num_positives,
        num_hard_negatives=args.num_hard_negatives,
    )
