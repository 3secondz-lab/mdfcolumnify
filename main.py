import argparse

from mdfcolumnify.mdfcolumnify import MdfColumnify
from mdfcolumnify.util.mdfhandler import MDF

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

    # assert not (
    #     not args.dbc and args.dst == "parquet"
    # ), "Raw message cannot be stored in parquet format.\n \
    #     Add --dbc option or choose other destination format."

    data = MdfColumnify(input=args.input, fmt=args.fmt, dbc_list=args.dbc)
    data.columnify(output=args.output, dst=args.dst, export=True)
