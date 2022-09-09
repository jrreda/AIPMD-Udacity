from PIL import Image
import numpy as np
import json
import torch
from torchvision import transforms

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path).resize((256,256))
    
    # centre crop
    width, height = img.size   # Get dimensions
    new_width, new_height = 224, 224
    
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    # convert colour channel from 0-255, to 0-1
    np_img = np.array(img)/255
    
    # normalize for model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean)/std
    
    # tranpose color channge to 1st dim
    np_img = np_img.transpose((2 , 0, 1))
    
    # convert to Float Tensor
    tensor = torch.from_numpy(np_img).type(torch.FloatTensor)
    
    return tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# Display an image along with the top 5 classes
def display_preditions(image_path, probs, class_names):
    """display the image & the model predictions"""

    # Plotting reults
    fig, (ax1, ax2) = plt.subplots(figsize=(5,6), nrows=2)

    # plot Image
    ax1.axis('off')
    imshow(process_image(image_path), ax=ax1)
    ax1.set_title(class_names[0])

    # plot barchart - top classes
    graph = ax2.barh(class_names, probs)

    # add percents on the graph
    i = 0
    for p in graph:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        plt.text(x+width/2,
                 y+height*0.45,
                 str(round(probs[i]*100, 3))+'%',
                 weight='bold'
                )
        i+=1

    return plt.show()


def mapping(model, top_p, top_class, category_names):
    # mapping from index to class
    mapping = {val: key for key, val in model.class_to_idx.items()}
    
    # convert tensors o lists
    probs, classes = top_p.tolist()[0], top_class.tolist()[0]
    probs, classes = probs, [mapping[c] for c in classes]
    
    # get names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[c] for c in classes]
    
    return probs, class_names
    
    
    
