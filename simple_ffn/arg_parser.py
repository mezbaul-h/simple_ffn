import argparse
import enum


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='simple_ffn',
        description='Simple FFN',
        epilog='(c) 2023',
    )

    parser.add_argument(
        '-f',
        '--filename',
        nargs=1,
        required=True,
    )

    return parser
