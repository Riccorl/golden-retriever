import argparse
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import zipfile

import huggingface_hub

from golden_retriever.common.utils import SAPIENZANLP_DATE_FORMAT, get_md5
from golden_retriever.common.logging import get_logger

logger = get_logger(level=logging.DEBUG)


def create_info_file(tmpdir: Path):
    logger.debug("Computing md5 of model.zip")
    md5 = get_md5(tmpdir / "model.zip")
    date = datetime.now().strftime(SAPIENZANLP_DATE_FORMAT)

    logger.debug("Dumping info.json file")
    with (tmpdir / "info.json").open("w") as f:
        json.dump(dict(md5=md5, upload_date=date), f, indent=2)


def zip_run(
    dir_path: Union[str, os.PathLike],
    tmpdir: Union[str, os.PathLike],
    zip_name: str = "model.zip",
) -> Path:
    logger.debug(f"zipping {dir_path} to {tmpdir}")
    # creates a zip version of the provided dir_path
    run_dir = Path(dir_path)
    zip_path = tmpdir / zip_name

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        # fully zip the run directory maintaining its structure
        for file in run_dir.rglob("*.*"):
            if file.is_dir():
                continue

            zip_file.write(file, arcname=file.relative_to(run_dir))

    return zip_path


def upload(
    model_dir: Union[str, os.PathLike],
    model_name: str,
    organization: Optional[str] = None,
    repo_name: Optional[str] = None,
    commit: Optional[str] = None,
    transformers: bool = False,
):
    token = huggingface_hub.HfFolder.get_token()
    if token is None:
        print(
            "No HuggingFace token found. You need to execute `huggingface-cli login` first!"
        )
        return

    repo_id = repo_name or model_name
    if organization is not None:
        repo_id = f"{organization}/{repo_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        api = huggingface_hub.hf_api.HfApi()
        repo_url = api.create_repo(
            token=token,
            repo_id=repo_id,
            exist_ok=True,
        )
        repo = huggingface_hub.Repository(
            str(tmpdir), clone_from=repo_url, use_auth_token=token
        )

        tmp_path = Path(tmpdir)
        if transformers:
            # if the user wants to upload a transformers model, we don't need to zip it
            # we just need to copy the files to the tmpdir
            logger.debug(f"Copying {model_dir} to {tmpdir}")
            os.system(f"cp -r {model_dir}/* {tmpdir}")
        else:
            # otherwise we zip the model_dir
            zip_run(model_dir, tmp_path)
            create_info_file(tmp_path)

        # this method automatically puts large files (>10MB) into git lfs
        repo.push_to_hub(commit_message=commit or "Automatic push from sapienzanlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir", help="The directory of the model you want to upload"
    )
    parser.add_argument("model_name", help="The model you want to upload")
    parser.add_argument(
        "--organization",
        help="the name of the organization where you want to upload the model",
    )
    parser.add_argument(
        "--repo_name",
        help="Optional name to use when uploading to the HuggingFace repository",
    )
    parser.add_argument(
        "--commit", help="Commit message to use when pushing to the HuggingFace Hub"
    )
    parser.add_argument(
        "--transformers", action="store_true", help="Upload as a transformers compatible model"
    )
    return parser.parse_args()


def main():
    upload(**vars(parse_args()))


if __name__ == "__main__":
    main()
