# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

#     Basic usage: python predict.py /path/to/image checkpoint
#     Options:
#         Return top K most likely classes: python predict.py ./flowers/test/1/image_06764.jpg ./vgg13_checkpoint.pth -k 3
#         Use a mapping of categories to real names: python predict.py ./flowers/test/1/image_06764.jpg ./vgg13_checkpoint.pth -cat cat_to_name.json -k 2
#         Use GPU for inference: python predict.py ./flowers/test/1/image_06764.jpg ./vgg13_checkpoint.pth -k 3 -cuda True

import argparse, sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from model_functions import load_checkpoint, predict
from utility_functions import display_preditions


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    # Basic usage
    parser.add_argument('image_path', default="./flowers/test/10/image_07090.jpg", type=str)
    parser.add_argument('checkpoint', default="./vgg13_checkpoint.pth", type=str)

    # Options
    parser.add_argument('-k', '--top_k', default=5, type=int)
    parser.add_argument('-cat', '--category_names', default="./cat_to_name.json", type=str)
    parser.add_argument('-cuda', '--gpu', default="True", type=bool)

    # Parse and print the results
    args = parser.parse_args()

    # load a checkpoint
    model = load_checkpoint(args.checkpoint)

    # predict image class, get the probabilities and classes
    probs, classes = predict(args.image_path, model, args.category_names, args.gpu, args.top_k)
    print(probs, classes)
    print(f"The flower name is '{classes[np.argmax(probs)]}' with {round(np.max(probs)*100, 2)}% confedince")
