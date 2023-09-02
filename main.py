import cv2
import numpy as np
import pickle

class mnist:
    def __init__(self, image):
        new_image = cv2.imread(image)
        resized_image = cv2.resize(new_image, (28, 28))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        normalized_image = binary_image / 255.0
        flattened_image = normalized_image.flatten()
        self.output_image = flattened_image

    def predict(self):
        with open('model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        x = loaded_model.predict(self.output_image.reshape(-1, 28, 28, 1))
        mx = 0
        for i in range(len(x[0])):
            if x[0][i] > mx:
                mx = x[0][i]
                ans = i
        print(ans)

if __name__ == "__main__":
    new = mnist(image='08u77.png')
    new.predict()
