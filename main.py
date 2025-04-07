from test_model import test_model
from train_model import train_and_save_model


def main():
    print("start learning")
    train_and_save_model()
    print("start testing")
    test_model()


if __name__ == "__main__":
    main()
