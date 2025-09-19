# tprofiler

[![PyPI](https://img.shields.io/pypi/v/tprofiler)](https://pypi.org/project/tprofiler/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tprofiler)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/b98a0fd1468c4858abf2a355bc9b4039/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/b98a0fd1468c4858abf2a355bc9b4039/raw/comments.json)

[![Code Test](https://github.com/deepghs/tprofiler/workflows/Code%20Test/badge.svg)](https://github.com/deepghs/tprofiler/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/deepghs/tprofiler/workflows/Package%20Release/badge.svg)](https://github.com/deepghs/tprofiler/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/deepghs/tprofiler/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/deepghs/tprofiler)

[![Discord](https://img.shields.io/discord/1157587327879745558?style=social&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FTwdHJ42N72)](https://discord.gg/TwdHJ42N72)
![GitHub Org's stars](https://img.shields.io/github/stars/deepghs)
[![GitHub stars](https://img.shields.io/github/stars/deepghs/tprofiler)](https://github.com/deepghs/tprofiler/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/deepghs/tprofiler)](https://github.com/deepghs/tprofiler/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/deepghs/tprofiler)
[![GitHub issues](https://img.shields.io/github/issues/deepghs/tprofiler)](https://github.com/deepghs/tprofiler/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/deepghs/tprofiler)](https://github.com/deepghs/tprofiler/pulls)
[![Contributors](https://img.shields.io/github/contributors/deepghs/tprofiler)](https://github.com/deepghs/tprofiler/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/deepghs/tprofiler)](https://github.com/deepghs/tprofiler/blob/master/LICENSE)

Useful utilities for huggingface

## Quick Start

To get started with `tprofiler`, you can install it using PyPI:

```shell
pip install tprofiler

```

Alternatively, you can install it from the source code:

```shell
git clone https://github.com/deepghs/tprofiler.git
cd tprofiler
pip install .

```

Verify the installation by checking the version:

```shell
tprofiler -v

```

If Python is not available in your local environment, we recommend downloading the pre-compiled executable version from
the [releases](https://github.com/deepghs/tprofiler/releases).

## Upload Data

Upload data to repositories using the following commands:

```shell
# Upload a single file to the repository
tprofiler upload -r your/repository -i /your/local/file -f file/in/your/repo

# Upload files in a directory as an archive file to the repository
# More formats of archive files are supported
# See: https://deepghs.github.io/tprofiler/main/api_doc/archive/index.html
tprofiler upload -r your/repository -i /your/local/directory -a archive/file/in/your/repo.zip

# Upload files in a directory as a directory tree to the repository
tprofiler upload -r your/repository -i /your/local/directory -d dir/in/your/repo

```

You can achieve the same using the Python API:

```python
from tprofiler.operate import upload_file_to_file, upload_directory_as_archive, upload_directory_as_directory

# Upload a single file to the repository
upload_file_to_file(
    local_file='/your/local/file',
    repo_id='your/repository',
    file_in_repo='file/in/your/repo'
)

# Upload files in a directory as an archive file to the repository
# More formats of archive files are supported
# See: https://deepghs.github.io/tprofiler/main/api_doc/archive/index.html
upload_directory_as_archive(
    local_directory='/your/local/directory',
    repo_id='your/repository',
    archive_in_repo='archive/file/in/your/repo.zip'
)

# Upload files in a directory as a directory tree to the repository
upload_directory_as_directory(
    local_directory='/your/local/directory',
    repo_id='your/repository',
    path_in_repo='dir/in/your/repo'
)
```

Explore additional options for uploading:

```shell
tprofiler upload -h

```

## Download Data

Download data from repositories using the following commands:

```shell
# Download a single file from the repository
tprofiler download -r your/repository -o /your/local/file -f file/in/your/repo

# Download an archive file from the repository and extract it to the given directory
# More formats of archive files are supported
# See: https://deepghs.github.io/tprofiler/main/api_doc/archive/index.html
tprofiler download -r your/repository -o /your/local/directory -a archive/file/in/your/repo.zip

# Download files from the repository as a directory tree
tprofiler download -r your/repository -o /your/local/directory -d dir/in/your/repo

```

Use the Python API for the same functionality:

```python
from tprofiler.operate import download_file_to_file, download_archive_as_directory, download_directory_as_directory

# Download a single file from the repository
download_file_to_file(
    local_file='/your/local/file',
    repo_id='your/repository',
    file_in_repo='file/in/your/repo'
)

# Download an archive file from the repository and extract it to the given directory
# More formats of archive files are supported
# See: https://deepghs.github.io/tprofiler/main/api_doc/archive/index.html
download_archive_as_directory(
    local_directory='/your/local/directory',
    repo_id='your/repository',
    file_in_repo='archive/file/in/your/repo.zip',
)

# Download files from the repository as a directory tree
download_directory_as_directory(
    local_directory='/your/local/directory',
    repo_id='your/repository',
    dir_in_repo='dir/in/your/repo'
)
```

Explore additional options for downloading:

```shell
tprofiler download -h

```

## List Files in Repository

List files in repositories

```shell
tprofiler ls -r your/repository -o /your/local/file -d subdir/in/repo
```

## Supported Formats

By default, we support the `zip` and `tar` formats, including `.zip`, `.tar`, `.tar.gz`, `.tar.bz2`, and `.tar.xz`.

If you require support for `rar` and `7z` files, install the extra dependencies using the following command:

```shell
pip install tprofiler[rar,7z]
```

**NOTE:** Creating RAR archive files is not supported. We use the [rarfile](https://github.com/markokr/rarfile) library,
which lacks the functionality for creating RAR files.

## How to Access Private Repositories

Simply configure the `HF_TOKEN` environment variable by using your HuggingFace access token.
Note that write permissions are required if you plan to upload any content.

## How to Use Hf-Transfer for Acceleration

If you are using the PyPI CLI, you need to install `tprofiler` with the following command:

```shell
pip install tprofiler[transfer]

```

If you are using a precompiled executable file, the transfer module is integrated inside; simply use it.

Enable Hf-Transfer acceleration by setting the environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `1`.

## How to Change the Temporary Directory

The temporary directory is utilized for storing partially downloaded files, consuming a considerable amount of disk
space.

If your disk, especially the C drive on Windows, does not have sufficient space, simply set the `TMPDIR` to designate
another directory as the temporary directory.
