from models import resnet, sequential, sequentialAdvanced, randomForestClassifier


class ResnetModel:
    @staticmethod
    def train():
        resnet.train_model.train_model()

    @staticmethod
    def test_model():
        resnet.test_model.test_model()


class SequentialModel:
    @staticmethod
    def train():
        sequential.train_model.train_model()


class SequentialAdvancedModel:
    @staticmethod
    def train():
        sequentialAdvanced.train_model.train_model()

class RandomForestClassifier:
    @staticmethod
    def train():
        randomForestClassifier.train_model.train_model()