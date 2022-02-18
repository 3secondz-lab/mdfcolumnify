import csv
import logging
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from asammdf import MDF as ASAMMDF
from asammdf.blocks.utils import (
    MDF2_VERSIONS,
    MDF3_VERSIONS,
    MDF4_VERSIONS,
    SUPPORTED_VERSIONS,
    MdfException,
    UniqueDB,
    components,
    csv_bytearray2hex,
    csv_int2hex,
    downcast,
    master_using_raster,
    validate_version_argument,
)

logger = logging.getLogger("mdfcolumnify")
formatter = logging.Formatter(
    fmt="{asctime} - {name:10s} [{levelname:^7s}] {message}",
    style="{",
    datefmt="%m/%d/%Y %H:%M:%S",
)
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)
LOCAL_TIMEZONE = datetime.now(timezone.utc).astimezone().tzinfo


class MDF(ASAMMDF):
    @staticmethod
    def export(self, fmt, filename=None, **kwargs):
        header_items = (
            "date",
            "time",
            "author_field",
            "department_field",
            "project_field",
            "subject_field",
        )
        if fmt != "pandas" and filename is None and self.name is None:
            message = (
                "Must specify filename for export"
                "if MDF was created without a file name"
            )
            logger.warning(message)
            return

        single_time_base = kwargs.get("single_time_base", False)
        raster = kwargs.get("raster", None)
        time_from_zero = kwargs.get("time_from_zero", False)
        use_display_names = kwargs.get("use_display_names", True)
        empty_channels = kwargs.get("empty_channels", "skip")
        format = kwargs.get("format", "5")
        oned_as = kwargs.get("oned_as", "row")
        reduce_memory_usage = kwargs.get("reduce_memory_usage", False)
        compression = kwargs.get("compression", "")
        time_as_date = kwargs.get("time_as_date", True)
        ignore_value2text_conversions = kwargs.get(
            "ignore_value2text_conversions", False
        )
        raw = bool(kwargs.get("raw", False))
        time_as_unix = kwargs.get("time_as_unix", True)

        if time_as_unix:
            time_as_date = True

        filename = Path(filename) if filename else self.name

        if fmt == "parquet":
            try:
                from pyarrow import parquet as pq
            except ImportError:
                logger.warning("pyarrow not found; export to parquet is unavailable")
                return

        elif fmt == "hdf5":
            try:
                from h5py import File as HDF5
            except ImportError:
                logger.warning("h5py not found; export to HDF5 is unavailable")
                return

        elif fmt not in ("csv",):
            raise MdfException(f"Export to {fmt} is not implemented")

        name = ""

        if self._callback:
            self._callback(0, 100)

        if single_time_base or fmt == "parquet":
            df = self.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                use_display_names=use_display_names,
                empty_channels=empty_channels,
                reduce_memory_usage=reduce_memory_usage,
                ignore_value2text_conversions=ignore_value2text_conversions,
                raw=raw,
                time_as_unix=time_as_unix,
            )
            units = OrderedDict()
            comments = OrderedDict()
            used_names = UniqueDB()

            dropped = {}

            groups_nr = len(self.groups)
            for i, grp in enumerate(self.groups):
                if self._terminate:
                    return

                for ch in grp.channels:

                    if use_display_names:
                        channel_name = ch.display_name or ch.name
                    else:
                        channel_name = ch.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if hasattr(ch, "unit"):
                        unit = ch.unit
                        if ch.conversion:
                            unit = unit or ch.conversion.unit
                    else:
                        unit = ""
                    comment = ch.comment

                    units[channel_name] = unit
                    comments[channel_name] = comment

                if self._callback:
                    self._callback(i + 1, groups_nr * 2)

        if fmt == "hdf5":
            filename = filename.with_suffix(".hdf")

            if single_time_base:

                with HDF5(str(filename), "w") as hdf:
                    # header information
                    group = hdf.create_group(str(filename))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = self.header[item].replace(b"\0", b"")

                    count = len(df.columns)

                    for i, channel in enumerate(df):
                        samples = df[channel]
                        unit = units.get(channel, "")
                        comment = comments.get(channel, "")

                        if samples.dtype.kind == "O":
                            if isinstance(samples[0], np.ndarray):
                                samples = np.vstack(samples)
                            else:
                                continue

                        if compression:
                            dataset = group.create_dataset(
                                channel, data=samples, compression=compression
                            )
                        else:
                            dataset = group.create_dataset(channel, data=samples)
                        unit = unit.replace("\0", "")
                        if unit:
                            dataset.attrs["unit"] = unit
                        comment = comment.replace("\0", "")
                        if comment:
                            dataset.attrs["comment"] = comment

                        if self._callback:
                            self._callback(i + 1 + count, count * 2)

            else:
                with HDF5(str(filename), "w") as hdf:
                    group = hdf.create_group(str(filename))

                    if self.version in MDF2_VERSIONS + MDF3_VERSIONS:
                        for item in header_items:
                            group.attrs[item] = self.header[item].replace(b"\0", b"")

                    groups_nr = len(self.virtual_groups)
                    for i, (group_index, virtual_group) in enumerate(
                        self.virtual_groups.items()
                    ):
                        channels = self.included_channels(group_index)[group_index]

                        if not channels:
                            continue

                        names = UniqueDB()
                        if self._terminate:
                            return

                        if len(virtual_group.groups) == 1:
                            comment = self.groups[
                                virtual_group.groups[0]
                            ].channel_group.comment
                        else:
                            comment = "Virtual group i"

                        group_name = r"/" + f"ChannelGroup_{i}"
                        group = hdf.create_group(group_name)

                        group.attrs["comment"] = comment

                        master_index = self.masters_db.get(group_index, -1)

                        if master_index >= 0:
                            group.attrs["master"] = (
                                self.groups[group_index].channels[master_index].name
                            )

                        channels = [
                            (None, gp_index, ch_index)
                            for gp_index, channel_indexes in channels.items()
                            for ch_index in channel_indexes
                        ]

                        if not channels:
                            continue

                        channels = self.select(channels, raw=raw)

                        for j, sig in enumerate(channels):
                            if use_display_names:
                                name = sig.display_name or sig.name
                            else:
                                name = sig.name
                            name = name.replace("\\", "_").replace("/", "_")
                            name = names.get_unique_name(name)
                            if reduce_memory_usage:
                                sig.samples = downcast(sig.samples)
                            if compression:
                                dataset = group.create_dataset(
                                    name, data=sig.samples, compression=compression
                                )
                            else:
                                dataset = group.create_dataset(
                                    name, data=sig.samples, dtype=sig.samples.dtype
                                )
                            unit = sig.unit.replace("\0", "")
                            if unit:
                                dataset.attrs["unit"] = unit
                            comment = sig.comment.replace("\0", "")
                            if comment:
                                dataset.attrs["comment"] = comment

                        if self._callback:
                            self._callback(i + 1, groups_nr)

        elif fmt == "csv":
            fmtparams = {
                "delimiter": kwargs.get("delimiter", ",")[0],
                "doublequote": kwargs.get("doublequote", True),
                "lineterminator": kwargs.get("lineterminator", "\r\n"),
                "quotechar": kwargs.get("quotechar", '"')[0],
            }

            quoting = kwargs.get("quoting", "MINIMAL").upper()
            quoting = getattr(csv, f"QUOTE_{quoting}")

            fmtparams["quoting"] = quoting

            escapechar = kwargs.get("escapechar", None)
            if escapechar is not None:
                escapechar = escapechar[0]

            fmtparams["escapechar"] = escapechar

            if single_time_base:
                filename = filename.with_suffix(".csv")
                message = f'Writing csv export to file "{filename}"'
                logger.info(message)

                if time_as_date:
                    index = (
                        pd.to_datetime(
                            df.index + self.header.start_time.timestamp(), unit="s"
                        )
                        .tz_localize("UTC")
                        .tz_convert(LOCAL_TIMEZONE)
                        .astype(str)
                    )
                    df.index = index
                    df.index.name = "timestamps"

                if time_as_unix:
                    df.index = pd.to_datetime(df.index).map(pd.Timestamp.timestamp)
                    df.index.name = "TimeStamp"

                if hasattr(self, "can_logging_db") and self.can_logging_db:

                    dropped = {}

                    for name_ in df.columns:
                        if name_.endswith("CAN_DataFrame.ID"):
                            dropped[name_] = pd.Series(
                                csv_int2hex(df[name_].astype("<u4") & 0x1FFFFFFF),
                                index=df.index,
                            )

                        elif name_.endswith("CAN_DataFrame.DataBytes"):
                            dropped[name_] = pd.Series(
                                csv_bytearray2hex(df[name_]), index=df.index
                            )

                    df = df.drop(columns=list(dropped))
                    for name, s in dropped.items():
                        df[name] = s

                with open(filename, "w", newline="") as csvfile:

                    writer = csv.writer(csvfile, **fmtparams)

                    names_row = [df.index.name, *df.columns]
                    writer.writerow(names_row)

                    if reduce_memory_usage:
                        vals = [df.index, *(df[name] for name in df)]
                    else:
                        vals = [
                            df.index.to_list(),
                            *(df[name].to_list() for name in df),
                        ]
                    count = len(df.index)

                    if self._terminate:
                        return

                    for i, row in enumerate(zip(*vals)):
                        writer.writerow(row)

                        if self._callback:
                            self._callback(i + 1 + count, count * 2)

            else:

                filename = filename.with_suffix(".csv")

                gp_count = len(self.virtual_groups)
                for i, (group_index, virtual_group) in enumerate(
                    self.virtual_groups.items()
                ):

                    if self._terminate:
                        return

                    message = f"Exporting group {i+1} of {gp_count}"
                    logger.info(message)

                    if len(virtual_group.groups) == 1:
                        comment = self.groups[
                            virtual_group.groups[0]
                        ].channel_group.comment
                    else:
                        comment = ""

                    if comment:
                        for char in r' \/:"':
                            comment = comment.replace(char, "_")
                        group_csv_name = (
                            filename.parent
                            / f"{filename.stem}.ChannelGroup_{i}_{comment}.csv"
                        )
                    else:
                        group_csv_name = (
                            filename.parent / f"{filename.stem}.ChannelGroup_{i}.csv"
                        )

                    df = self.get_group(
                        group_index,
                        raster=raster,
                        time_from_zero=time_from_zero,
                        use_display_names=use_display_names,
                        reduce_memory_usage=reduce_memory_usage,
                        ignore_value2text_conversions=ignore_value2text_conversions,
                        raw=raw,
                    )

                    if time_as_date:
                        index = (
                            pd.to_datetime(
                                df.index + self.header.start_time.timestamp(), unit="s"
                            )
                            .tz_localize("UTC")
                            .tz_convert(LOCAL_TIMEZONE)
                            .astype(str)
                        )
                        df.index = index
                        df.index.name = "timestamps"
                    if time_as_unix:
                        df.index = pd.to_datetime(df.index).map(pd.Timestamp.timestamp)
                        df.index.name = "TimeStamp"
                    with open(group_csv_name, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile, **fmtparams)

                        if hasattr(self, "can_logging_db") and self.can_logging_db:

                            dropped = {}

                            for name_ in df.columns:
                                if name_.endswith("CAN_DataFrame.ID"):
                                    dropped[name_] = pd.Series(
                                        csv_int2hex(df[name_] & 0x1FFFFFFF),
                                        index=df.index,
                                    )

                                elif name_.endswith("CAN_DataFrame.DataBytes"):
                                    dropped[name_] = pd.Series(
                                        csv_bytearray2hex(df[name_]), index=df.index
                                    )

                            df = df.drop(columns=list(dropped))
                            for name_, s in dropped.items():
                                df[name_] = s

                        names_row = [df.index.name, *df.columns]
                        writer.writerow(names_row)

                        if reduce_memory_usage:
                            vals = [df.index, *(df[name] for name in df)]
                        else:
                            vals = [
                                df.index.to_list(),
                                *(df[name].to_list() for name in df),
                            ]

                        for i, row in enumerate(zip(*vals)):
                            writer.writerow(row)

                    if self._callback:
                        self._callback(i + 1, gp_count)

        elif fmt == "parquet":
            filename = filename.with_suffix(".parquet")
            df.to_parquet(filename)

        else:
            message = (
                'Unsopported export type "{}". '
                'Please select "csv", "excel", "hdf5", "mat" or "pandas"'
            )
            message.format(fmt)
            logger.warning(message)

    @staticmethod
    def to_dataframe(
        self,
        channels=None,
        raster=None,
        time_from_zero=False,
        empty_channels="skip",
        keep_arrays=False,
        use_display_names=False,
        time_as_date=True,
        reduce_memory_usage=False,
        raw=False,
        ignore_value2text_conversions=False,
        use_interpolation=True,
        only_basenames=False,
        interpolate_outwards_with_nan=False,
        time_as_unix=True,
    ):
        if time_as_unix:
            time_as_date = True
        if channels is not None:
            mdf = self.filter(channels)

            result = mdf.to_dataframe(
                raster=raster,
                time_from_zero=time_from_zero,
                empty_channels=empty_channels,
                keep_arrays=keep_arrays,
                use_display_names=use_display_names,
                time_as_date=time_as_date,
                reduce_memory_usage=reduce_memory_usage,
                raw=raw,
                ignore_value2text_conversions=ignore_value2text_conversions,
                use_interpolation=use_interpolation,
                only_basenames=only_basenames,
                interpolate_outwards_with_nan=interpolate_outwards_with_nan,
            )

            mdf.close()
            return result

        df = {}

        self._set_temporary_master(None)

        if raster is not None:
            try:
                raster = float(raster)
                assert raster > 0
            except (TypeError, ValueError):
                if isinstance(raster, str):
                    raster = self.get(raster).timestamps
                else:
                    raster = np.array(raster)
            else:
                raster = master_using_raster(self, raster)
            master = raster
        else:
            masters = {index: self.get_master(index) for index in self.virtual_groups}

            if masters:
                master = reduce(np.union1d, masters.values())
            else:
                master = np.array([], dtype="<f4")

            del masters

        idx = np.argwhere(np.diff(master, prepend=-np.inf) > 0).flatten()
        master = master[idx]

        used_names = UniqueDB()
        used_names.get_unique_name("timestamps")

        groups_nr = len(self.virtual_groups)

        for group_index, (virtual_group_index, virtual_group) in enumerate(
            self.virtual_groups.items()
        ):
            if virtual_group.cycles_nr == 0 and empty_channels == "skip":
                continue

            channels = [
                (None, gp_index, ch_index)
                for gp_index, channel_indexes in self.included_channels(
                    virtual_group_index
                )[virtual_group_index].items()
                for ch_index in channel_indexes
                if ch_index != self.masters_db.get(gp_index, None)
            ]

            signals = [
                signal
                for signal in self.select(
                    channels, raw=True, copy_master=False, validate=False
                )
            ]

            if not signals:
                continue

            group_master = signals[0].timestamps

            for sig in signals:
                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(
                            len(master)
                            if virtual_group.cycles_nr == 0
                            else virtual_group.cycles_nr,
                            dtype=sig.samples.dtype,
                        )
                        sig.timestamps = (
                            master if virtual_group.cycles_nr == 0 else group_master
                        )

            if not raw:
                if ignore_value2text_conversions:

                    for signal in signals:
                        conversion = signal.conversion
                        if conversion:
                            samples = conversion.convert(signal.samples)
                            if samples.dtype.kind not in "US":
                                signal.samples = samples
                else:
                    for signal in signals:
                        if signal.conversion:
                            signal.samples = signal.conversion.convert(signal.samples)

            for s_index, sig in enumerate(signals):
                sig = sig.validate(copy=False)

                if len(sig) == 0:
                    if empty_channels == "zeros":
                        sig.samples = np.zeros(
                            len(master)
                            if virtual_group.cycles_nr == 0
                            else virtual_group.cycles_nr,
                            dtype=sig.samples.dtype,
                        )
                        sig.timestamps = (
                            master if virtual_group.cycles_nr == 0 else group_master
                        )

                signals[s_index] = sig

            if use_interpolation:
                same_master = np.array_equal(master, group_master)

                if not same_master and interpolate_outwards_with_nan:
                    idx = np.argwhere(
                        (master >= group_master[0]) & (master <= group_master[-1])
                    ).flatten()

                cycles = len(group_master)

                signals = [
                    signal.interp(master, self._integer_interpolation)
                    if not same_master or len(signal) != cycles
                    else signal
                    for signal in signals
                ]

                if not same_master and interpolate_outwards_with_nan:
                    for sig in signals:
                        sig.timestamps = sig.timestamps[idx]
                        sig.samples = sig.samples[idx]

                group_master = master

            signals = [sig for sig in signals if len(sig)]

            if signals:
                diffs = np.diff(group_master, prepend=-np.inf) > 0
                if np.all(diffs):
                    index = pd.Index(group_master, tupleize_cols=False)

                else:
                    idx = np.argwhere(diffs).flatten()
                    group_master = group_master[idx]

                    index = pd.Index(group_master, tupleize_cols=False)

                    for sig in signals:
                        sig.samples = sig.samples[idx]
                        sig.timestamps = sig.timestamps[idx]
            else:
                index = pd.Index(group_master, tupleize_cols=False)

            size = len(index)
            for k, sig in enumerate(signals):
                sig_index = (
                    index
                    if len(sig) == size
                    else pd.Index(sig.timestamps, tupleize_cols=False)
                )

                # byte arrays
                if len(sig.samples.shape) > 1:

                    if use_display_names:
                        channel_name = sig.display_name or sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    df[channel_name] = pd.Series(
                        list(sig.samples),
                        index=sig_index,
                    )

                # arrays and structures
                elif sig.samples.dtype.names:
                    for name, series in components(
                        sig.samples,
                        sig.name,
                        used_names,
                        master=sig_index,
                        only_basenames=only_basenames,
                    ):
                        df[name] = series

                # scalars
                else:
                    if use_display_names:
                        channel_name = sig.display_name or sig.name
                    else:
                        channel_name = sig.name

                    channel_name = used_names.get_unique_name(channel_name)

                    if reduce_memory_usage and sig.samples.dtype.kind not in "SU":
                        sig.samples = downcast(sig.samples)

                    df[channel_name] = pd.Series(
                        sig.samples, index=sig_index, fastpath=True
                    )

            if self._callback:
                self._callback(group_index + 1, groups_nr)

        strings, nonstrings = {}, {}

        for col, series in df.items():
            if series.dtype.kind == "S":
                strings[col] = series
            else:
                nonstrings[col] = series

        df = pd.DataFrame(nonstrings, index=master)

        for col, series in strings.items():
            df[col] = series

        df.index.name = "timestamps"

        if time_as_date:
            new_index = np.array(df.index) + self.header.start_time.timestamp()
            new_index = pd.to_datetime(new_index, unit="s")

            df.set_index(new_index, inplace=True)
        elif time_from_zero and len(master):
            df.set_index(df.index - df.index[0], inplace=True)

        if time_as_unix:
            df.index = pd.to_datetime(df.index).map(pd.Timestamp.timestamp)
            df.index.name = "TimeStamp"

        return df
