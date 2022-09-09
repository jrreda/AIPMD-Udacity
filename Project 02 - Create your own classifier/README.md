# Developing an AI application  - Flowers Recognition

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' alt="Flowers" width="80%;"/>

When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.


There are 2 python executables:

- [train.py](https://github.com/jrreda/AIPND-Udacity/blob/main/Project%2002%20-%20Create%20your%20own%20classifier/train.py)
- [predict.py](https://github.com/jrreda/AIPND-Udacity/blob/main/Project%2002%20-%20Create%20your%20own%20classifier/predict.py)

## Training the classifier
`train.py` will train the classifier. The user will need to specify one **mandatory argument**:
- `'data_dir'` contating the path to the training data directory as `str`.

**Optional arguments:**
- `-s`: the saving directory.
- `-a`: the user can choose which architecture to use for the neural network. The default architecture is `VGG13`
- `-lr`: sets the Learning rate for gradient descent as `float`: default is 0.001.
- `-e`: specifies the number of epochs as `int`. Set to 3 by default.
- `-hu`: a `list` of `int`s specifying how many neurons the hidden-layers will contain.
- `-do`: a `list` of `floats`s specifying the number of Dropouts used bey each hiiden layer.
- `-cuda`: the user should specifify `True` to use GPU for training if it is available. The model will use the CPU otherwise.

## Using the classifier
`predict.py` will accept an image as input and will output a probability ranking of predicted flower species. The **mandatory argument** are:
- `image_path`: the path to the input image we what to know (predict) it's name.
- `checkpoint`: the checkpoint path we what to know load the traind model.

**The options are:**
- `-k`: let's the user specify the numer of top K-classes to output. Default is 5.
- `-cat`: allows user to provide path of JSON file mapping categories to names.
- `-cuda`: the user should specifify `True` to use GPU for prediction if it is available. The model will use the CPU otherwise.
