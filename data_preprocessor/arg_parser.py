import argparse


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='data_preprocessor',
        description='Preprocessor for FFN',
        epilog='(c) 2023',
    )

    parser.add_argument(
        '-f',
        '--filename',
        help='Source filename',
        nargs=1,
        required=True,
    )

    return parser
