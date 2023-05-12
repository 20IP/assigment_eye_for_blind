import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob


def show_img_desc(list_img, df_text, num=5):
    random_img = random.choices(list_img, k=num)
    for name in random_img:
        description = df_text[df_text['image'] == os.path.basename(name)]['caption'].tolist()
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        for txt in description:
            print(txt)