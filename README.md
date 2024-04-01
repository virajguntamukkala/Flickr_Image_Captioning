# Flickr Image Captioning

## About
**Provided an image, generate a descriptive caption for that image.** 

This project implements a image captioning model using the Flickr8K dataset. As an enthusiast of Computer Vision and Natural Language Processing, I am building this project as it is a combination of CV and NLP.


## Details

### Dataset
The dataset Flickr8K can be found at [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k). The Flick8K is benchmark collection for sentence-based image description and search, consisting of around 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and event
  

### Model

 ![Model
  ](https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/03-advanced/image_captioning/png/model.png)
The model is built on encoder-decoder architecture with attention. 

The encoder is a Convolutional Neural Network (CNN) that is trained to identify the features of the input image. We employ transfer learning by using a ResNet50 network that has been pre-trained on ImageNet. This encoder will process the input image and produces the encoded image vectors that capture the essential features of the image.  

The encoder passes these features to the decoder, which is a Recurrent Neural Network (LSTM),  which will decode the features to a sequence of words. The decoder generates the caption word-by-word in an autoregressive manner, where each predicted word is fed back into the decoder to predict the next word along with the encoded image features. In this decoder, we also apply local attention using Bahdanau Attention. So, as the Decoder generates each word of the output sequence, the Attention module will guide the decoder to to focus on the most relevant part of the image for generating that word.

### Metrics

To evaluate the model's performance, I will use the automated BiLingual Evaluation Understudy (BLEU) evaluation metric. BLEU is a commonly used metric for NLP problems as it intuitive and quick to calculate.  BLEU measures the similarity between the generated captions and the reference (ground truth) captions, by looking at the overlap of n-grams (sequences of n words). There are different variants of BLEU score (BLEU-1, BLEU-2, BLEU-3, BLEU-4) that consider 1-gram, 2-gram, 3-gram, and 4-gram overlaps respectively. BLEU score ranges from 0 to 1, with 1 indicating a perfect match between the generated and reference captions. BLEU is a corpus-level metric, meaning it is calculated across the entire set of generated and reference captions, not on individual captions. However, while BLEU is widely used, it has some limitations, such as not considering semantic similarity and not handling paraphrasing well.

![BLEU Formula](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUsAAACYCAMAAABatDuZAAABX1BMVEX///8AAADayrnb6fb8///y8vJUVFRra2v8/Pz///jz/P+xsbHGxsZhYWHt3s02NjajuMk0AACOorb78OKnk33w/v+ampoAABzo6Oj///SgjHTCsJ2Jb1TPvqwAAAspTWwAIEmRkZEiAADE1OVfQRf36NcnAAC4pZD+9uuuwNPt+P9uhpw5DQCarsHa0cgcAABVYm7AytPS0tKIn7MAACPi8PxvVDMUAAD49PC+vr56k6qqqqohISF7e3t/hpkACj7c3NwAACp7bFxAW3XW2dvS3uhCGwBVb4iYjH9wdX9nTzhRLgBJQj6CjJeDeG2or7lZUEkAK1AYPVtzZ1tMWGQyQFG0qqAALk5CSFBbQCNFJgAUJzk2KBXl1MRddo2Bc21SRz5aOQQZHinBt65GV2xma3NrTy9BNSu3r6cAJDl4Y06XnaRLNiaXfmVrZW1JIAAqIRopOEksGgA5MUIuLi55Cw5PAAAM3klEQVR4nO1di1sS2xbfm0QZQDRnzoTAeMQkSGtSShoiih4Wmoq9zDzR6WRaJw/33rr3///uXnvPwLwAsQFk3L/vE8ZhHnt+s9Zej/1CiIODg4ODg4ODg4ODg4NjdKA9fjLsIvgFdbzdz8vH2Zf2tJ83OSeo42nbnvjjm95dPosvwVc6/xive3fVc4pnz2071KqD3V9AbGOMfGaXEdr08A2dT8TwVfsuwUsuGV7MtPslH/b6XsNDgOmgGd5yGYKPy225VLe2fcNmn7ms7OAwUuVbK/KqebcmyzVjW93dWDL/FC2tI/XEsxIMEP2WyywGsbPJZXZlCRWuN/9Vd1dahQhl8FoSv/SuBINDv7lUXLhMvkJIxGumPfHXb5pWPvCyfjXgeY09CAyBy5zzlkS1d97qmp7YqDl/HgkMgUvJTYH39mtjbOvFyEZhQ+Cy4CBLqE41ZVF0OmmjgsFwec3MZeod+Yi17LpW3TcZ+eS8dzcfMFy4jOF73l1fgtAxhs2iWMdvIuWmHGo7byzuUmHZu5v/Gog/ty/LE7R4lSrZLo+vdjzByWU8ms/nvSpPPJ+PhjW4YtC0Uz5pepTSii1MNx84ZGQwhL0BfBtB1XMTNjtKmYtccuig/JE6CaopEUM19R4vdThe4Vy2hc5lAihiXKY6siXdGUy5RhE6lwePxgwuD+6MtT9coJUBhytE/CAez7/+Yx3pXCYwC3y1UsRAqVXdf381pHKOAkS8EY3KH6ghJ9uRtGEytWgLersBEja/dJDZCw9dxzdBGpmOd8CtjiZe7dSoELt+ioNGHDqXAfz8FFxq7+0NFOYrwYXi5TaCWzUyuHudvITRhs5lHX+0cRkbnzQw3kodbrZPJJTJn3aI3bkUmm2Xub62Yg4VzFdH3yF06yqXYMfbHSJS/Z+ds3EZYl8BmoAIQozyZ+cqVz8h9CvhTG63c/DWHwSr+G0pIn/aJ1QGJfwq3s22JB61+UGhwmvjMpR9wvI4afInyE+TZDvbMa+TZDnfb8XNH2cPCjJD8dyCwVAwGAy2tkNdTlAs+W0TJGpU7HKZPKlVyQmzIM27S6hKDlI6ZpXq+1CxJpcRetZZSfam2hQE0O0pzgfaxuMNup9ymUsz1BAqMEKS5Mes7gQETtHu/Ve3ymZvalST6Sa05dJVLgU9tgcVL+i6rZyCy/ZtugCt+mMY9aHncONShZxYktWX2Ho086EyQB8RUa2IutSXWnVuyaVNV/iWbv6v7ZhTwShUKa4jdSSl1Mml8HoaSa9QBlQ4dIhXzYIpMW+K+pTi1zf0if/s1FeA2D+XNt3Am3VDrOM7tgSmpuJLATzf63OcBzi5PCB8JZ4w/7INzC7l7IOO13dr01XmwV9j901+tqt3/WFsWunqzJ1HOLiUficPe2epFSU6UTf/stfB/CJXLmepV9U0zXtbVoWWfNOmK9zaPiyfUL0+bOcQpk1qr3axGi5cZh/ajjncOjFd8f3I5lvsXMbwdDNCOY1X191/dXCZcsatph5aOf+06bI+hKpnCQu3Nl3ImeZs8qzubjA2FbvUjg4c9eX7uWK07J27l2WNJeZ8lIIf5PecHdkqjMvUuWnT7RkOLrXqRM27CiseiRTDWjESiZpyG2q53D7reTi6CVGXfsEcZ8WNj0O9fXbcwP7oNy4HcFs/kqNXKN0zxhynhfZ9ZHs8cnBwcHBwcHBwcHBwcHBwcHBwcAwV52d44EjicEKW02XydxRWP/EBOL+EnUswxOk3unH8+7BLM9qAXo2Uy8w6SnEufxmUSwITl6PRafscosWl+AnG6iDxpPT1AQp9+3B97xZvhu0JTS7/XkV1fBXlvkDHxXsoho/W2w8k4XCDWccFfBNJK5FI8dZDwud16B/g34FzfYCFy7sz6HgGBuuEGJf1doNyONxg5fIm+ksfTaPLJeeyB9i5TNEqMs+5PAPsXCpgzHPbJh33bN4b32NT7xd5PA9duJdh5C3eurLExuEqmNhy3GHsN0cLWrRYLMLMz2qxGA1G6fZh+SSMNLK/EicfYfVkf9il9BE2hl0A3yCe5j6mV+B5TQ4ODg4ODg4ODg4ODg4ODg4ODt9is4/dSP4Z3ek3zoKGh0uBOJC7UN0qlHd9vXxgJJc+Oxty5rZ6rQhzsamRVQ+nEUv1U+zPFwqWR2Vrmf3Lyxu4Lo/mS9Stk4ZnxpfJ03s7UaB0UbrzvrDOMP1tFoeR4q0gzV6Q6ajqNjMro9RN8uEtGhdDMAvWjjhEu+svkdcLuMYuRI1pX4AwQB76WdFz1/3F6M49eHpkr1idnxIyVtD19jYXoU/5ok3FV+Dji+e3uQiTIeasq3Jq+TysshRvc/Qv4Jr//fWBLeKX8v9IxkSnZem8RFbvJe1jLJocv3w8aIa3g+QCvq8wBWzyVUrYgrSnd9JXy/IxMpYnnMB4km2F4iWPuZzFfvcwbdPj3se4GT2WOqxmcRb8e1BhpCoDSk97PjHfOlGL0s2eWgSS1losTlTbGCgTmuh0Yrzaq9EqDGwGewWUrfL+757T+VmQLPXnW4hUUngJqbiXWa0lm0tUJGQaNqej7Qkmel3FN9VmPUDvEaPKJiz0rAjMPuboUCEJ/I6eJmJu2NtiyhiPn+rM5JXT34UiMbBmH8Yllaz4zv0aDTy0w/9sEK2dCiP161uietqey0JYuq9xDI9GuczgHuZhTjki76lTGvCeubSrQP+gc3k8FwZ6DG24DAHe4RqpbKgRhEWLXk8wTOq06lw+e4iaXFqUXCMvowZrHtRQ4GgVVar3xK+fm5JbcNR65ipTvwILK8eQUDTtZVzuyTtH0FikleWJWrxCLxaK55FacghhFg+qbZdxyfSz1WhwAzwWLWwslwFchgzohzAuaa2J6GpaDYs5gVxFhsTcCvzNkOdcONrIP256QotOgwBVpnm0TC5Nl3b8+Ru5junKlMvGbbj1GhKukV+ufYlCuUPq3eVv0R2HRicH5qzH8BtZ/oBpAqAllzeIXIrwOltc2hHAR7L8k5Ej4ZK8a1nGDSXu5fMVTGLh1B0NUkBoEWqAg3n9ZxcuocrcsuyQnjPGkGRqrwQuM5SwhedsWZqEca1rX8JsAK0Fg+SSyeUPi44/W5Hlr1DNtLjc1dd+2LLoeBJDmkwy1X667i/OVJ4+rRAlnb32iF51Ea6UNaTGjUu0ZdNy5R0SI6Ch5mw7cCnR5RJTc2N0qRbpofEo62AMh80lYeO2XS6VbnJJi9iA3Jlkyh/UWfX0rBnXJHRy4UrNpU5duazgkuX/+kOURkSLD81qC1w26LtLkM/Nj0i4obNHGx2d4+MHzmUADIfOZYjVlwI8QVcuk1A9SM5czHvqZFUIITMNKpiMS2P9TTcuQ9gW8cQeqVfRi2mRlLCyuxbYovaYyiU1zaDbmQ05bZiWRXh9qedIq97MvW75Z0OQy+uG7RGv6rYHRUh5KSUZl/SAzmUKvly4lKAmFbZR7gjN3gUaF8HKJ4xk+rFL08GU3cEU8Tasy0k1/L/qpQQtKasv6Yqd95DwuXU07VgQg0oksVqLtfyzbJPLSp8NOo17tD3MNJAURIOeE3fhtSanoYDk/sKESzMMteBalToAKacLJxzg/YkrS2h3TZeM40dhJM4Z2lq44ghGJu/b94hQnsYrODL2dh3dWGP3DcObWkLiH2OEusl0Wl/XFGRevAt0faohqSWX2WbhFnoNmXoDi8fTEerJleh2DWlFthceXCun5RMXKlk8LkeCLB53RvRCdXyDRJbpVSR8k+V1tEjc/3Kz4ks5JDmNHcOHaScOutApyj5BGZogj5N7hukcgHQxw2p6YnycZYIaK2lWZHgFB62X2/LVuy2iODJYtKS+HIFdCbfaeioRx9nHl1B2xhmoJ1mjEe2EtGAYcOUeEudbxzT8l1i3cpm11QpRszc0WbGfPEuq1+MZp3qwBrg61J6x5tspXEXKl9abSc0NKrcxMFy2ROuK1bhWsClMLFkXiAaEgu4zCOzht+XyBO3P1WhaMxJ1hkwHH/utFybMqGL2w+sWlzpIIp4pA6dNGFGE8sU8Fbt8sRh1T9bd8PvSVyI2dXIL4T42+Mxi3zeQL5iSSuXJCTP+V2p/Wu/IWDs1+BGF/nZVb2FgnRqGB8n7blhtbuSMCvyG+qCi5GO/mx7obHC77W+qh9N6zfq+qwFCHWbZ1pR5727jWIzaj1CMOIX41YLVD8/Me3ebxqBs3DAhGMqXfCKmrQkcL7lcuAAqjlBC7xipnKwKd5CaZtgOe8rlBZlqWtTbWlPLSLF2+fCQy4LfO2bpSLEk+8IaSl1HakTHmJdcZi7K7MgiNbHQmnA3jrS4DuDSs95UF0UsEcqCW5ScRrl9S59/1y43Z0JsYN2yho+DPi+Ocdnv3atNEPtrZROn1vD/A5iOYI4ed5XTAAAAAElFTkSuQmCC)

## Results
||Train|Val|Test|
|--|--|--|--|
BLEU-1| 0.5604897566671818 | 0.5328947368421053 | 0.5314397224631396 |
BLEU-2| 0.3941290328132996 | 0.3555287497311401 | 0.35849421587786245 |
BLEU-3| 0.26985389021930145 | 0.22903707628833264 | 0.23216074406872433 |
BLEU-4| 0.17824254119305746 | 0.14229152531864633 | 0.14726022432329325 |

BLEU score is a precision based measure and it ranges from 0 to 1. The closer the value is to 1, the better the prediction. It is not possible to achieve a value close 1. A score of 0.6 is usually considered as high-quality translations.  These results are on par with publications and other works in image captioning. However, there is room for improvement. 

## Next Steps

 - [ ] Implement Beam Search
 - [ ] Try other models
	 - [ ] Encoders (ResNet101, etc)
	 - [ ] Decoders (Vision Transformers)
 - [ ] Train using larger Flickr30K dataset
 - [ ] Deploy model using AWS/GCP with a website built with Flask

## Acknowledgments
- Implementation for Bahdanau Attention: [Link](https://github.com/makarovartyom/Image-Captioning-with-Attention/blob/c31f81769f15b52e5c082aced2e682b028830b12/model.py#L168)
