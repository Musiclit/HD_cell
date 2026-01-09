import configparser
from datetime import datetime

import numpy as np
from scipy.stats import ranksums, ttest_ind, ttest_rel, wilcoxon

from lotr import dataset_folders
from lotr.default_vals import RESULTS_LOG_FILE, RESULTS_NDIGITS
from lotr.file_utils import get_figures_location

test_dict = dict(
    ttest_ind=ttest_ind, ttest_rel=ttest_rel, wilcoxon=wilcoxon, ranksums=ranksums
)


def _round_to_ndig(x):
    if x == 0:
        return x
    log = np.floor(np.log10(abs(x)))
    rounded = round(x, -int(log) + RESULTS_NDIGITS - 1)
    return rounded


def _format_number(x):
    try:
        return str([_round_to_ndig(v) for v in x])
    except TypeError:
        return str(_round_to_ndig(x))


class ResultsLogger:
    """A class to log all statistical results from running the analysis and generating the figures
    in the notebooks to a single file in the figure folder.
    """

    def __init__(self, log_filename=None):
        if log_filename is None:
            log_filename = get_figures_location() / RESULTS_LOG_FILE

        self.log_filename = log_filename

        if self.log_filename.exists():
            configprs = self.get_config_parser()
            # TODO maybe there's a smarter way:
            dataset_raw_read = configprs["log_info"]["dataset"]
            self.dataset = set(dataset_raw_read.split("'")[1::2])
        else:
            self.dataset = set([d.name for d in dataset_folders])

        self.create_reslog_file()

    def get_config_parser(self):
        configparser_obj = configparser.ConfigParser()
        configparser_obj.read(self.log_filename)
        return configparser_obj

    def create_reslog_file(self):
        if not self.log_filename.exists():
            config = configparser.ConfigParser()
            config["log_info"] = {
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": self.dataset,
            }

            with open(self.log_filename, "w") as configfile:
                config.write(configfile)

    def check_dataset_consistency(self, dataset_to_check):
        assert all([d in self.dataset for d in dataset_to_check])

    def add_entry(
        self, name, values, fids, units="", moment="mean", save_vals=False, n_items=None
    ):
        # Check number of values match number of fish:
        assert len(values) == len(fids)

        # If we pass objects (eg Paths) take just the name:
        if type(fids[0]) is not str:
            fids = [f.name for f in fids]
        fids = list(fids)

        self.check_dataset_consistency(fids)

        if set(fids) == self.dataset:
            fids = "all"

        if moment == "mean":
            value = np.mean(values)
            interval = np.std(values)
        elif moment == "median":
            value = np.median(values)
            interval = np.percentile(values, [25, 75])
        elif moment == "minmax":
            value = np.median(values)
            interval = (np.min(values), np.max(values))
        else:
            raise ValueError(f"Invalid moment specified: {moment}")

        cfgparser = self.get_config_parser()
        if name not in cfgparser.sections():
            cfgparser.add_section(name)

        cfgparser.set(name, "value", _format_number(value))
        cfgparser.set(name, "interval", _format_number(interval))
        cfgparser.set(name, "n_fish", str(len(fids)))
        cfgparser.set(name, "moment", moment)
        cfgparser.set(name, "units", units)

        # Save raw values if specified:
        if save_vals:
            cfgparser.set(name, "values", ",".join([str(v) for v in values]))

        # Save n items if required:
        if n_items is not None:
            cfgparser.set(name, "n_items", n_items)

        # Save fish ids or just write "all":
        fids_log_val = "all" if (set(fids) == self.dataset) else str(fids)
        cfgparser.set(name, "fids", fids_log_val)

        with open(self.log_filename, "w") as configfile:
            cfgparser.write(configfile)

    def add_statcomparison(self, name1, name2, vals1, vals2, test):
        s, p = test_dict[test](vals1, vals2)

        cfgparser = self.get_config_parser()

        assert all([n in cfgparser.sections() for n in [name1, name2]])

        name = f"{name1} vs {name2}"
        if name not in cfgparser.sections():
            cfgparser.add_section(name)

        cfgparser.set(name, "s", _format_number(s))
        cfgparser.set(name, "p_val", _format_number(p))
        cfgparser.set(name, "test", test)

        with open(self.log_filename, "w") as configfile:
            cfgparser.write(configfile)
