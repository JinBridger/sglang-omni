#!/usr/bin/env python3
"""Download an OSS model directory using runtime environment variables."""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import oss2
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
REQUIRED_ENV = (
    "OSS_ENDPOINT",
    "OSS_REGION",
    "OSS_BUCKET_NAME",
    "OSS_ACCESS_KEY_ID",
    "OSS_ACCESS_KEY_SECRET",
    "OSS_MODEL_PATH",
    "LOCAL_MODEL_PATH",
)
DOWNLOAD_ATTEMPTS = 3
PART_SIZE_BYTES = 16 * 1024 * 1024


def download_model() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    endpoint = os.environ["OSS_ENDPOINT"].strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    bucket_name = os.environ["OSS_BUCKET_NAME"].strip()
    source = os.environ["OSS_MODEL_PATH"].strip()
    if source.startswith("oss://"):
        parsed = urlsplit(source)
        if parsed.netloc != bucket_name:
            raise ValueError("OSS_MODEL_PATH bucket does not match OSS_BUCKET_NAME")
        source = parsed.path
    prefix = source.strip("/")
    if not prefix:
        raise ValueError("OSS_MODEL_PATH must name a model directory within the bucket")
    prefix += "/"

    destination = Path(os.environ["LOCAL_MODEL_PATH"]).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    state_directory = destination / ".oss-download"
    state_directory.mkdir(exist_ok=True)
    bucket = oss2.Bucket(
        oss2.AuthV4(
            os.environ["OSS_ACCESS_KEY_ID"],
            os.environ["OSS_ACCESS_KEY_SECRET"],
        ),
        endpoint,
        bucket_name,
        region=os.environ["OSS_REGION"].strip(),
        connect_timeout=60,
        enable_crc=True,
    )
    store = oss2.ResumableDownloadStore(root=str(state_directory))
    downloaded_count = 0
    skipped_count = 0

    # SDK checkpoints must not be written by concurrent container startups.
    with (state_directory / "download.lock").open("a") as download_lock:
        fcntl.flock(download_lock, fcntl.LOCK_EX)
        for remote_file in oss2.ObjectIteratorV2(bucket, prefix=prefix):
            if remote_file.key.endswith("/"):
                continue
            relative_name = remote_file.key[len(prefix) :]
            relative_path = PurePosixPath(relative_name)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.parts[0] == ".oss-download"
            ):
                raise ValueError(f"Unsafe object path: {relative_name!r}")
            target = (destination / relative_name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(
                    f"Object path escapes LOCAL_MODEL_PATH: {relative_name!r}"
                )

            source_id = f"{endpoint}/{bucket_name}/{remote_file.key}"
            state_id = hashlib.sha256(source_id.encode()).hexdigest()
            metadata_path = state_directory / f"{state_id}.json"
            expected_metadata = {"etag": remote_file.etag, "size": remote_file.size}
            if target.is_file() and metadata_path.is_file():
                try:
                    saved_metadata = json.loads(metadata_path.read_text())
                except (ValueError, OSError):
                    saved_metadata = None
                local_stat = target.stat()
                if (
                    saved_metadata
                    == {
                        **expected_metadata,
                        "mtime_ns": local_stat.st_mtime_ns,
                    }
                    and local_stat.st_size == remote_file.size
                ):
                    LOGGER.info(f"Already downloaded: {relative_name}")
                    skipped_count += 1
                    continue

            target.parent.mkdir(parents=True, exist_ok=True)
            partial_path = state_directory / f"{state_id}.part"
            LOGGER.info(f"Downloading {relative_name} ({remote_file.size:,} bytes)")
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                try:
                    oss2.resumable_download(
                        bucket,
                        remote_file.key,
                        str(partial_path),
                        store=store,
                        multiget_threshold=PART_SIZE_BYTES,
                        part_size=PART_SIZE_BYTES,
                        num_threads=4,
                    )
                    break
                except oss2.exceptions.OssError as error:
                    retryable = (
                        isinstance(error, oss2.exceptions.RequestError)
                        or error.status == 429
                        or error.status >= 500
                    )
                    if not retryable or attempt == DOWNLOAD_ATTEMPTS:
                        raise
                    LOGGER.warning(
                        f"Download interrupted; retrying ({attempt}/{DOWNLOAD_ATTEMPTS})"
                    )
                    time.sleep(2**attempt)

            if partial_path.stat().st_size != remote_file.size:
                raise RuntimeError(
                    f"Object size changed during download: {relative_name}"
                )
            os.replace(partial_path, target)
            metadata_path.write_text(
                json.dumps(
                    {
                        **expected_metadata,
                        "mtime_ns": target.stat().st_mtime_ns,
                    }
                )
            )
            downloaded_count += 1
            LOGGER.info(f"Downloaded: {relative_name}")

    if downloaded_count + skipped_count == 0:
        raise ValueError(
            "No model files found under OSS_MODEL_PATH; check the directory prefix"
        )
    LOGGER.info(
        f"Model ready at {destination}: {downloaded_count} downloaded, {skipped_count} unchanged"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Load a .env file without overriding existing environment variables",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        if arguments.env_file is not None:
            if not arguments.env_file.is_file():
                raise FileNotFoundError(
                    f"Environment file not found: {arguments.env_file}"
                )
            load_dotenv(arguments.env_file, override=False, interpolate=False)
        download_model()
    except oss2.exceptions.OssError as error:
        # Raw SDK errors can include signed requests; report only diagnostic IDs.
        LOGGER.error(
            f"OSS download failed: {type(error).__name__}, status={error.status}, "
            f"code={error.code}, request_id={error.request_id}"
        )
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
