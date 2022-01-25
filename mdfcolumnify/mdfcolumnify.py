import argparse
import bz2
import glob
import logging
import os

import pandas as pd
from asammdf import MDF, Signal

logger = logging.getLogger("mdfcolumnify")


class MdfColumnify(object):
    def __init__(
        self,
        input: str | None = None,
        fmt: str = "mf4",
        dbc_list: list[str] | None = None,
    ):
        if input is not None:
            self._mdf = self.concatenate_input(input, fmt)
        else:
            self._mdf = MDF()
        self.dbc_list = dbc_list

    def __setattr__(self, name, value):
        if name == "_mdf":
            super().__setattr__(name, value)
        else:
            setattr(self._mdf, name, value)

    def __getattr__(self, name):
        return getattr(self._mdf, name)

    def __enter__(self):
        return self

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._mdf)))

    def __exit__(self):
        if self._mdf is not None:
            try:
                self.close()
            except:
                pass
        self._mdf = None

    def __del__(self):
        if self._mdf is not None:
            try:
                self.close()
            except:
                pass
        self._mdf = None

    def concatenate_input(self, src, fmt="mf4"):
        src = os.path.join(os.getcwd(), src)
        if os.path.isfile(src):
            input_files = [src]
        elif os.path.isdir(src):
            input_files = sorted(glob.glob(src + "/*." + fmt))
            logger.info(f"Concatenating {len(input_files)} Log Files in {src}")
            logger.info(f"{input_files}")
        else:
            logger.error(f"{src} is not a valid file")

        if fmt == "bz2":
            for file in input_files:
                mdf = MDF.concatenate(mdf, MDF(bz2.BZ2File(file, "rb")))

        else:
            mdf = MDF.concatenate(input_files)
        return mdf

    def filter_invalid_channels(self):
        channel_list = []
        channel_iterator = self.iter_channels()

        for channel in channel_iterator:
            if len(channel.timestamps):
                channel_list.append(channel.name)
        logger.info(f"{len(channel_list)} Valid channels in files")
        logger.info(f"{channel_list}")
        return self.filter(channel_list)

    def extract_bus(self, dbc_list=None):
        if not dbc_list:
            dbc_list = self.dbc_list
        dbc_tuple_list = []
        if len(dbc_list):
            for bus in range(len(dbc_list)):
                dbc_tuple_list.append((dbc_list[bus], 0))
            #     dbc_tuple_list.append((dbc_list[bus], bus))

        return self.extract_bus_logging({"CAN": dbc_tuple_list})

    def save_files(self, output, dst):
        try:
            self.export(output, dst)
        except Exception as e:
            logger.error(f"Error occurs when saving files: {e}")

    def columnify(self, output: str, dst: str = "csv", dbc_list: list[str] = None):
        if dbc_list:
            self.dbc_list = dbc_list
            filtered = self.extract_bus()
        elif self.dbc_list:
            filtered = self.extract_bus()
        else:
            if dst == "parquet":
                logger.error(
                    f"Raw message cannot be stored in parquet format.\n"
                    + "specify dbc file or choose other destination format."
                )
                return False
            else:
                filtered = self.filter_invalid_channels()

        if not filtered.info()["groups"]:
            logger.warning(f"Empty data, nothing to export {filtered.info()}")
            return False

        filtered.export(dst, output)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deserialize .mf4 file into (.csv | .h5 | .parquet) formats"
    )
    parser.add_argument("input", help="path to MF4 source")
    parser.add_argument("output", help="path to converted destination")
    parser.add_argument("--dbc", help="path to dbc file for can frames", nargs="+")
    parser.add_argument(
        "--fmt", help="input source format", default="mf4", choices=["mf4", "bz2"]
    )
    parser.add_argument(
        "--dst",
        help="output format ['csv', 'hdf5', 'parquet']",
        default="csv",
        choices=["csv", "hdf5", "parquet"],
    )
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    assert not (
        not args.dbc and args.dst == "parquet"
    ), "Raw message cannot be stored in parquet format.\n \
        Add --dbc option or choose other destination format."

    data = MdfColumnify(input=args.input, fmt=args.fmt, dbc_list=args.dbc)
    data.columnify(output=args.output, dst=args.dst)
