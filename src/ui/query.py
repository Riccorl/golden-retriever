import hydra
import omegaconf

from data.pl_data_modules import PLDataModule
from models.pl_modules import GoldenRetrieverPLModule


def run_ui(conf: omegaconf.DictConfig) -> None:
    pl_data_module: PLDataModule = hydra.utils.instantiate(
        conf.data.datamodule, _recursive_=False
    )
    # force setup to get labels initialized for the model
    pl_data_module.prepare_data()

    pl_module = GoldenRetrieverPLModule.load_from_checkpoint(conf.ui.checkpoint_path)

    dataloaders = (
        trainer.val_dataloaders
        if stage == "validation"
        else trainer.test_dataloaders
    )
    datasets = (
        trainer.datamodule.val_datasets
        if stage == "validation"
        else trainer.datamodule.test_datasets
    )

    # compute the context embeddings index for each dataloader
    for dataloader_idx, dataloader in enumerate(dataloaders):
        logger.log(f"Computing context embeddings for dataloader {dataloader_idx}")
        if datasets[dataloader_idx].contexts is not None:
            context_dataloader = DataLoader(
                datasets[dataloader_idx].contexts,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=lambda x: ModelInputs(
                    {
                        # this is a hack to normalize the batch structure
                        "contexts": trainer.datamodule.tokenizer(
                            x, padding=True, return_tensors="pt"
                        )
                    }
                ),
            )
        else:
            context_dataloader = dataloader

        context_embeddings, context_index = self.compute_context_embeddings(
            pl_module.model.context_encoder, context_dataloader, pl_module.device
        )


@hydra.main(config_path="../../conf", config_name="default")
def main(conf: omegaconf.DictConfig):
    run_ui(conf)


if __name__ == "__main__":
    main()