import re
import tempfile
from pathlib import Path
from shutil import copy

import ipynbname
import pooch
from tqdm import tqdm

from lotr.default_vals import DATASET_HASH, DATASET_URL


def get_dataset_location():
    """Handles finding the source data of the analysis.
    By default, tries to find the repo dataset_location.txt file.
    If not available, download test dataset from web (for CI).

    Returns
    -------
    dataset location

    """
    specification_txt = Path(__file__).parent.parent / "dataset_location.txt"
    if specification_txt.exists():
        with open(specification_txt, "r") as f:
            return Path(f.read())

    # If we did not find anything, download from zotero example dataset:
    data_pooch = pooch.create(
        path=pooch.os_cache("lotr"),
        base_url=DATASET_URL,
        registry={"sample_dataset.zip": DATASET_HASH},
    )

    unpack = pooch.Unzip(members=None)
    fnames = data_pooch.fetch("sample_dataset.zip", processor=unpack)
    print(fnames[0])
    # Ugly search of super parent folder, as the unzipping has to happen on files
    print(Path(f) for f in fnames if Path(f).name == "selected.h5")
    return ([Path(f) for f in fnames if Path(f).name == "selected.h5"])[
        0
    ].parent.parent.parent.parent


def get_figures_location():
    specification_txt = Path(__file__).parent.parent / "figures_location.txt"
    if specification_txt.exists():
        with open(specification_txt, "r") as f:
            return Path(f.read())

    return Path(tempfile.mkdtemp())


def folder_2_fid(folder):
    """take only actual fish id of date and fish number."""

    f_n = "".join(re.split(r"(\d+)", folder.name.split("_")[1])[:2])
    date = folder.name.split("_")[0]

    return f"{date}_{f_n}"


def mirror_all_subfolders(source_master_path, dest_master_path, file_patterns=None):
    dest_master_path.mkdir(exist_ok=True)

    # paths = [source_path / f.name for f in dest_master_path.glob("*_f*")]

    for path in tqdm(list(source_master_path.glob("*[0-9]_f[0-9]*"))):
        mirror_fish_folder(path, dest_master_path / path.name, file_patterns)


def mirror_fish_folder(source_path, dest_path, file_patterns=None, overwrite=False):
    if file_patterns is None:
        file_patterns = [
            "*behavior_log*",
            "*stimulus_log*",
            "*metadata.json",
            "data_from_suite2p_unfiltered.h5",
            "bouts_df.h5",
            "*selected*.h5",
            "filtered_traces.h5",
        ]
    # [

    dest_path.mkdir(exist_ok=True)

    for pattern in file_patterns:
        for file in source_path.glob(pattern):
            if overwrite or not (dest_path / file.name).exists():
                copy(file, dest_path / file.name)


def get_nb_name():
    try:
        return ipynbname.name()
    except IndexError:
        pass
