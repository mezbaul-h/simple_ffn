from simple_ffn.scalers import MinMaxScaler
from simple_ffn.settings import DATA_ROOT_DIR, DATASET_FILE_PREFIX
from simple_ffn.utils import read_dataset_csv


class Dataset:
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()
        self.x_train = None
        self.x_validation = None
        self.x_test = None
        self.y_train = None
        self.y_validation = None
        self.y_test = None

    def _lazy_initialization(self):
        self.feature_scaler.load_params(DATA_ROOT_DIR / "feature_scaler_params.json")
        self.output_scaler.load_params(DATA_ROOT_DIR / "output_scaler_params.json")

        self.x_train, self.y_train = read_dataset_csv(DATA_ROOT_DIR / f"{DATASET_FILE_PREFIX}.train.csv")
        self.x_validation, self.y_validation = read_dataset_csv(
            DATA_ROOT_DIR / f"{DATASET_FILE_PREFIX}.validation.csv"
        )
        self.x_test, self.y_test = read_dataset_csv(DATA_ROOT_DIR / f"{DATASET_FILE_PREFIX}.test.csv")

    def process(self):
        self._lazy_initialization()

        return self.x_train, self.x_validation, self.x_test, self.y_train, self.y_validation, self.y_test
