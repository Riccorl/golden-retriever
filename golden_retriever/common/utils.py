import importlib.util
import json
import logging
import os
import shutil
import tarfile
import tempfile
from ast import Dict
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union
from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile

import huggingface_hub
import requests
import tqdm
from filelock import FileLock

from golden_retriever.common.logging import get_logger

# name constants
WEIGHTS_NAME = "weights.pt"
ONNX_WEIGHTS_NAME = "weights.onnx"
CONFIG_NAME = "config.yaml"
LABELS_NAME = "labels.json"

# path constants
SAPIENZANLP_CACHE_DIR = os.getenv("SAPIENZANLP_CACHE_DIR", Path.home() / ".sapienzanlp")
SAPIENZANLP_DATE_FORMAT = "%Y-%m-%d %H-%M-%S"


logger = get_logger()


def is_package_available(package_name: str) -> bool:
    """
    Check if a package is available.

    Args:
        package_name (`str`): The name of the package to check.
    """
    return importlib.util.find_spec(package_name) is not None


def load_json(path: Union[str, Path]) -> Any:
    """
    Load a json file provided in input.

    Args:
        path (`Union[str, Path]`): The path to the json file to load.

    Returns:
        `Any`: The loaded json file.
    """
    with open(path, encoding="utf8") as f:
        return json.load(f)


def dump_json(document: Any, path: Union[str, Path], indent: Optional[int] = None):
    """
    Dump input to json file.

    Args:
        document (`Any`): The document to dump.
        path (`Union[str, Path]`): The path to dump the document to.
        indent (`Optional[int]`): The indent to use for the json file.

    """
    with open(path, "w", encoding="utf8") as outfile:
        json.dump(document, outfile, indent=indent)


def get_md5(path: Path):
    """
    Get the MD5 value of a path.
    """
    import hashlib

    with path.open("rb") as fin:
        data = fin.read()
    return hashlib.md5(data).hexdigest()


def file_exists(path: Union[str, os.PathLike]) -> bool:
    """
    Check if the file at :obj:`path` exists.

    Args:
        path (:obj:`str`, :obj:`os.PathLike`):
            Path to check.

    Returns:
        :obj:`bool`: :obj:`True` if the file exists.
    """
    return Path(path).exists()


def dir_exists(path: Union[str, os.PathLike]) -> bool:
    """
    Check if the directory at :obj:`path` exists.

    Args:
        path (:obj:`str`, :obj:`os.PathLike`):
            Path to check.

    Returns:
        :obj:`bool`: :obj:`True` if the directory exists.
    """
    return Path(path).is_dir()


def is_remote_url(url_or_filename: Union[str, Path]):
    """
    Returns :obj:`True` if the input path is an url.

    Args:
        url_or_filename (:obj:`str`, :obj:`Path`):
            path to check.

    Returns:
        :obj:`bool`: :obj:`True` if the input path is an url, :obj:`False` otherwise.

    """
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def url_to_filename(resource: str, etag: str = None) -> str:
    """
    Convert a `resource` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the resources's, delimited
    by a period.
    """
    resource_bytes = resource.encode("utf-8")
    resource_hash = sha256(resource_bytes)
    filename = resource_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def download_resource(
    url: str,
    temp_file: BinaryIO,
    headers=None,
):
    """
    Download remote file.
    """

    if headers is None:
        headers = {}

    r = requests.get(url, stream=True, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        desc="Downloading",
        disable=logger.level in [logging.NOTSET],
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def download_and_cache(
    url: Union[str, Path],
    cache_dir: Union[str, Path] = None,
    force_download: bool = False,
):
    if cache_dir is None:
        cache_dir = SAPIENZANLP_CACHE_DIR
    if isinstance(url, Path):
        url = str(url)

    # check if cache dir exists
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # check if file is private
    headers = {}
    try:
        r = requests.head(url, allow_redirects=False, timeout=10)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            hf_token = huggingface_hub.HfFolder.get_token()
            if hf_token is None:
                raise ValueError(
                    "You need to login to HuggingFace to download this model "
                    "(use the `huggingface-cli login` command)"
                )
            headers["Authorization"] = f"Bearer {hf_token}"

    etag = None
    try:
        r = requests.head(url, allow_redirects=True, timeout=10, headers=headers)
        r.raise_for_status()
        etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
        # We favor a custom header indicating the etag of the linked resource, and
        # we fallback to the regular etag header.
        # If we don't have any of those, raise an error.
        if etag is None:
            raise OSError(
                "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
            )
        # In case of a redirect,
        # save an extra redirect on the request.get call,
        # and ensure we download the exact atomic version even if it changed
        # between the HEAD and the GET (unlikely, but hey).
        if 300 <= r.status_code <= 399:
            url = r.headers["Location"]
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        # Actually raise for those subclasses of ConnectionError
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Otherwise, our Internet connection is down.
        # etag is None
        pass

    # get filename from the url
    filename = url_to_filename(url, etag)
    # get cache path to put the file
    cache_path = cache_dir / filename

    # the file is already here, return it
    if file_exists(cache_path) and not force_download:
        logger.info(
            f"{url} found in cache, set `force_download=True` to force the download"
        )
        return cache_path

    cache_path = str(cache_path)
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if file_exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        temp_file_manager = partial(
            tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
        )

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise, you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(
                f"{url} not found in cache or `force_download` set to `True`, downloading to {temp_file.name}"
            )
            download_resource(url, temp_file, headers)

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url}  # , "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def from_cache(
    url_or_filename: Union[str, Path],
    cache_dir: Union[str, Path] = None,
    force_download: bool = False,
) -> Path:
    """

    Args:
        url_or_filename:
        cache_dir:
        force_download:

    Returns:

    """
    if cache_dir is None:
        cache_dir = SAPIENZANLP_CACHE_DIR

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = download_and_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
        )
    elif file_exists(url_or_filename):
        logger.info(f"{url_or_filename} is a local path or file")
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(
            f"unable to parse {url_or_filename} as a URL or as a local path"
        )

    if dir_exists(output_path) or (
        not is_zipfile(output_path) and not tarfile.is_tarfile(output_path)
    ):
        return Path(output_path)

    # Path where we extract compressed archives
    # for now it will extract it in the same folder
    # maybe implement extraction in the sapienzanlp folder
    # when using local archive path?
    logger.info("Extracting compressed archive")
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

    # already extracted, do not extract
    if (
        os.path.isdir(output_path_extracted)
        and os.listdir(output_path_extracted)
        and not force_download
    ):
        return Path(output_path_extracted)

    # Prevent parallel extractions
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        os.makedirs(output_path_extracted)
        if is_zipfile(output_path):
            with ZipFile(output_path, "r") as zip_file:
                zip_file.extractall(output_path_extracted)
                zip_file.close()
        elif tarfile.is_tarfile(output_path):
            tar_file = tarfile.open(output_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError(
                f"Archive format of {output_path} could not be identified"
            )

    # remove lock file, is it safe?
    os.remove(lock_path)

    return Path(output_path_extracted)
