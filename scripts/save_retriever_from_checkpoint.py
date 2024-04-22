from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule
from goldenretriever.pytorch_modules.model import GoldenRetriever


model = GoldenRetriever(
    question_encoder="intfloat/e5-base-v2",
)
pl_module = GoldenRetrieverPLModule.load_from_checkpoint(
    "/root/golden-retriever/wandb/run-20240412_102023-ds9v63nc/files/checkpoints/checkpoint-validate_recall@100_0.9915-epoch_02.ckpt", model=model
)

pl_module.model.save_pretrained(
    "models/aida-e5-base-topics-from-blink-1M-32words-windows"
)
