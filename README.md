# Eye for Blinder
> Deep Learning application in helping the blind and visually impaired can know the content inside the photo / photo


## Table of Contents
1. [General Info](#general-information)
2. [Technologies Used](#technologies-used)
3. [Dataset Infomation](#dataset-infomation)
4. [Conclusion](#conclusion)
5. [Contact](#contact)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Research project on applying convolutional newral network to find features in images, combining Attwention and RNN networks to convert images into text + audio through Google API.
The purpose of supporting the blind + the visually impaired can hear the description inside the picture

## Technologies Used
tensorflow:  2.7.0
sklearn:  0.23.2
gtts:  2.3.2
pandas:  1.4.3
pillow:  9.2.0
#####_Main networks in the model_
- Pretrained of InceptionV3
- Attention
- RNN

## Dataset Infomation
- data used is famous dataset : Flickr8k
The data contains 8091 images of all kinds
Each category consists of 5 descriptive sentences.
Preliminary assessment : The data is still insufficient and the complexity of the sentences is unsatisfactory for most outside cases.

## Conclusion
- In conclusion, the task of image captioning using the Flickr8k dataset involves generating descriptive captions for images. The process includes leveraging a pre-trained InceptionV3 model for image feature extraction, an attention mechanism to focus on relevant image regions, and an RNN-based decoder for generating captions.

- Dataset Preparation: The Flickr8k dataset, consisting of images and corresponding captions, is used for training and evaluation. The images are preprocessed and fed into the InceptionV3 model to extract image features. The captions are tokenized and prepared for training the model.

- Model Architecture: The model architecture includes an encoder-decoder framework. The encoder utilizes the pre-trained InceptionV3 model to extract features from the images. The decoder consists of an attention mechanism that attends to specific image regions while generating captions. An RNN-based decoder, GRU is used to generate sequential words in the captions.

- Training: The model is trained using the encoded image features and the tokenized captions. The attention mechanism helps the decoder focus on relevant image regions during caption generation. The training involves optimizing the model parameters using techniques like teacher forcing and gradient descent.

- Evaluation: After training, the model is evaluated using a separate test set from the Flickr8k dataset. The evaluation involves generating captions for test images and comparing them with the ground truth captions. Metrics BLEU score can be used to evaluate the quality of the generated captions.

- By combining the power of pre-trained image models, attention mechanisms, and RNN-based decoders, the model can effectively learn to generate relevant and descriptive captions for images in the Flickr8k dataset.

- Overall, this approach provides a powerful solution for the image captioning task, allowing machines to understand and describe the content of images accurately.
- Results from 5 test images (not included in Flickr8k) also show some content of the image, although not very accurate, but could be better if we have more data, context or build more string modle can. can cover a lot of information. The above things can improve the Model more in the future

## Contact

Created by : </br>Pham Van Thai: phamthai.ats@gmail.com</br>Feel free to contact us!
- We do not have any constraints about License on the use of our results. You can use it for free.