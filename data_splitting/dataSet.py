import os
import random

import numpy as np
from skimage.io import imread

from utils.config import images_folder
from skimage.util import img_as_ubyte
class dataSet:

    trainingSet = []
    validationSet = []
    @staticmethod
    def extractDataset():
        processed_data = os.path.join(images_folder, "processed_frames")
        cases = os.listdir(processed_data)
        inputDir = os.path.join(images_folder, "processed_frames")
        class1 = []
        class2 = []

        for case in cases:
            frames = os.listdir(os.path.join(inputDir, case))
            for frame in frames:
                image = imread(os.path.join(inputDir, case, frame))
                image = np.array(img_as_ubyte(image))

                if case.find("plaman-normal") != -1:
                    class1.append(image)
                else:
                    class2.append(image)



        return class1, class2


    def getData(self):
        class1, class2 = self.extractDataset()
        finalLength = min(len(class1), len(class2))

        class1 = random.sample(class1, finalLength)
        class2 = random.sample(class2, finalLength)
        trainLength = int(self.percentage/100 * finalLength)
        validationLength = finalLength - trainLength

        chooseClass1 = random.sample(range(finalLength), trainLength)
        chooseClass2 = random.sample(range(finalLength), trainLength)

        self.trainingSet = []
        self.trainingSet.extend([(np.stack([class1[i]] * 3, -1), 1) for i in chooseClass1])
        self.trainingSet.extend([(np.stack([class2[i]] * 3, -1), 0) for i in chooseClass2])
        random.shuffle(self.trainingSet)

        self.validationSet = []
        for i in range(finalLength):
            if i not in chooseClass1:
                self.validationSet.append((np.stack([class1[i]] * 3, -1), 1))

        for i in range(finalLength):
            if i not in chooseClass2:
                self.validationSet.append((np.stack([class2[i]] * 3, -1), 0))

        random.shuffle(self.validationSet)




    def __init__(self, percentage):
        self.percentage = percentage
        self.getData()

