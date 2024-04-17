

from goldenretriever.lightning_modules.pl_modules import GoldenRetrieverPLModule


pl_module = GoldenRetrieverPLModule.load_from_checkpoint()

pl_module.model.save_pretrained("/home/ric/Projects/golden-retriever/wandb/run-20240412_102023-ds9v63nc")