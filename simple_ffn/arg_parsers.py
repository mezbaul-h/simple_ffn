"""
Argument Parsers Module

This module provides argument parsers for the main script.
"""
import argparse


def make_main_arg_parser():
    """
    Create and configure the argument parser for the main script.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="simple_ffn",
        description="Simple FFN",
        epilog="(c) 2023",
    )

    parser.add_argument(
        "-e",
        "--num-epochs",
        nargs=1,
        required=True,
        type=int,
    )

    parser.add_argument(
        "-H",
        "--hidden-size",
        default=[2],
        nargs=1,
        type=int,
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        nargs=1,
        required=True,
        type=float,
    )

    parser.add_argument(
        "-m",
        "--momentum",
        nargs=1,
        required=True,
        type=float,
    )

    return parser
