import os
import sys

from keras.engine.saving import load_model

import accuracy

from accent_detector.helper import get_wav, to_mfcc
from trainmodel import segment_one

DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10  # 35#250


class ModelLoader:

    def __init__(self):
        self.folder_path = "..\models"
        self.model_name = "model1.h5"

    def _get_model_path(self):
        return os.path.join(self.folder_path, self.model_name)

    def load_model(self):
        model = load_model(self._get_model_path())
        return model


class SoundPredictor:

    def __init__(self, path=None):
        self.model = ModelLoader().load_model()
        self.csv_file = "bio_data.csv"

    def predict(self, file_path):

        mfcc = to_mfcc(get_wav(file_path, base_folder="../audio"))
        mfcc = segment_one(mfcc)
        prediction = accuracy.predict_class_audio(mfcc, self.model)

        return prediction


if __name__ == '__main__':

    folder_path = sys.argv[1]
    if not folder_path:
        folder_path = "../audio"

    predictor = SoundPredictor()
    for i in range(1, 20):
        file_path = os.path.join(folder_path, "arabic" + str(i))
        print(predictor.predict(file_path))
    # print(predictor.predict("arabic6"))