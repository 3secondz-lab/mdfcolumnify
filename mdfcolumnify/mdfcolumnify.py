import bz2
import glob
import logging
import os

from .util.mdfhandler import MDF

logger = logging.getLogger("mdfcolumnify")
formatter = logging.Formatter(
    fmt="{asctime} - {name:10s} [{levelname:^7s}] {message}",
    style="{",
    datefmt="%m/%d/%Y %H:%M:%S",
)
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


class MdfColumnify(object):
    _mdf = None
    _abs_time = []

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
            input_files += sorted(glob.glob(src + "/*." + fmt.upper()))
            logger.info(f"Concatenating {len(input_files)} Log Files in {src}:")
            logger.info(f"{input_files}")
        else:
            logger.error(f"{src} is not a valid file")

        if fmt == "bz2":
            try:
                for file in input_files:
                    mdf = MDF.concatenate(mdf, MDF(bz2.BZ2File(file, "rb")))
            except Exception as e:
                logger.error(f"{src} is not valid MDF format: {e}")
        else:
            for file in input_files:
                _tmp = MDF(file)
                self._abs_time.append((file, _tmp.header.abs_time))
            mdf = MDF.concatenate(input_files)
        return mdf

    def filter_invalid_channels(self):
        channel_list = []
        channel_iterator = self.iter_channels()

        for channel in channel_iterator:
            if len(channel.timestamps):
                channel_list.append(channel.name)
        logger.info(f"{len(channel_list)} Valid channels in files:\n{channel_list}")
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
            self.export(output, dst, time_as_date=True)
        except Exception as e:
            logger.error(f"Error occurs when saving files: {e}")

    def columnify(
        self,
        output: str | None = None,
        dst: str | None = None,
        dbc_list: list[str] | None = None,
        export: bool = False,
        time_as_unix: bool = True,
    ):
        if dbc_list:
            self.dbc_list = dbc_list
            filtered = self.extract_bus()
        elif self.dbc_list:
            filtered = self.extract_bus()
        else:
            filtered = self.filter_invalid_channels()

        if not filtered.info()["groups"]:
            logger.warning(f"Empty data, nothing to export:\n {filtered.info()}")
            return False
        if export:
            if not dst in ["csv", "hdf5", "parquet"]:
                logger.error(f"Destination file {output} with unknown extension {dst}")
                raise RuntimeError("Export format is not valid")
            if dst == "parquet":
                logger.warning(
                    f"Converting Raw message to parquet will add additional dependency:\n[ pandas, pyarrow ]"
                )
                filtered = MDF.to_dataframe(
                    filtered, use_interpolation=False
                ).to_parquet(output + ".parquet")
                import pyarrow.parquet as pq

                _tmp = pq.read_table(output + ".parquet")
                logger.warning(
                    f"First Row of Converted Pandas dataframe:\n{_tmp.to_pandas().iloc[0]}"
                )
                logger.warning(f"Rows without value will be filled with NaN")

            else:
                MDF.export(
                    filtered, dst, output, time_as_date=True, time_as_unix=time_as_unix
                )
        return filtered
