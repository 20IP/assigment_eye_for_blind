import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import re
import string
import os
import glob


def show_img(rows, cols, images):
    img_idx = random.choices(images, k = rows*cols+1)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*5, rows*5))
    for num in range(1, rows*cols+1):
        
        fig.add_subplot(rows, cols, num)
        idx = num - 1
        img = cv2.imread(img_idx[num])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, aspect='auto')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(os.path.basename(img_idx[num]), fontsize=12, labelpad=20)
    plt.show()


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
            
def standardize(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, f'[{re.escape(string.punctuation)}]', '')
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text.numpy().decode('utf-8')