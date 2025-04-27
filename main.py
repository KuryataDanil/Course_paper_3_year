from test_model import test_model
from train_model import train_and_save_model
import warnings

warnings.filterwarnings("ignore")

def main():
    train_and_save_model()
    test_model()


if __name__ == "__main__":
    main()
