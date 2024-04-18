from omegaconf import OmegaConf

class_dict = {
    "wandb_log_model": True,
    "wandb_online_mode": False,
    "wandb_watch": "all",
    "wandb_kwargs": None,
    "model_checkpointing": True,
    "checkpoint_dir": None,
    "checkpoint_filename": None,
    "save_top_k": 1,
    "save_last": False,
    "checkpoint_kwargs": None,
    "prediction_batch_size": 128,
    "max_hard_negatives_to_mine": 15,
    "hard_negatives_threshold": 0.0,
    "metrics_to_monitor_for_hard_negatives": None,
    "mine_hard_negatives_with_probability": 1.0,
    "seed": 42,
    "float32_matmul_precision": "medium",
    "retriever": None,
    "train_dataset": None,
    "train_batch_size": None,
    "train_dataset_kwargs": {},
    "val_dataset": None,
    "val_batch_size": None,
    "val_dataset_kwargs": {},
    "test_dataset": None,
    "test_batch_size": None,
    "test_dataset_kwargs": {},
    "num_workers": None,
    "optimizer": None,
    "lr": None,
    "weight_decay": None,
    "lr_scheduler": None,
    "num_warmup_steps": None,
    "loss": None,
    "callbacks": None,
}

# Convert the dictionary to an OmegaConf object
class_conf = OmegaConf.create(class_dict)

# Assuming class_conf is your OmegaConf object
yaml_string = OmegaConf.to_yaml(class_conf)

# Write the YAML string to a file
with open("config.yaml", "w") as f:
    f.write(yaml_string)
