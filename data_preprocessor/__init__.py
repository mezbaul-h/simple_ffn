from .arg_parser import make_arg_parser
from .data_preprocessor import DataPreprocessor


def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    source_filename = args.filename[0]
    data_pp = DataPreprocessor(
        source_filename=source_filename,
    )

    data_pp.sanitize_data()
    data_pp.train_test_split()
    data_pp.scale_data()
    data_pp.save_data()
    data_pp.save_scaler_params()
