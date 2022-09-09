from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import warnings
warnings.filterwarnings('ignore')
from utility_functions import process_image, imshow, mapping


# TODO: Define your transforms for the training, validation, and testing sets
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],     # mean
                                                                     [0.229, 0.224, 0.225])]),  # std
        'valid_transforms': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],     # mean
                                                                    [0.229, 0.224, 0.225])]),  # std
        'test_transforms': transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],    # mean
                                                                    [0.229, 0.224, 0.225])])  # std
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms']),
        'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms'])

    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=64),
        'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64)
    }

    return data_transforms, image_datasets, dataloaders

def structure_network(arch="vgg13", hidden_units=[4096, 1000], dropouts=[0.5, 0.3]):
    # TODO: Build and train your network
    model = eval(f"models.{arch}(pretrained=True)") # models.vgg13(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Assign the classifier to the Pre-trained model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units[0])),
        ('relu_1', nn.ReLU()),
        ('droput_1', nn.Dropout(dropouts[0])),
        ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
        ('relu_2', nn.ReLU()),
        ('droput_2', nn.Dropout(dropouts[1])),
        ('fc3', nn.Linear(hidden_units[1], 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
    try:
        model.classifier = classifier
    except:
        model.clf = classifier
    
    return model

def evaluate_network(model, device, dataloaders, criterion):
    """Evaluate the network using Valid or Test dataloaders."""
    loss = 0
    accuracy = 0
    model.eval()

    for inputs, labels in dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return loss, accuracy

def save_checkpoint(save_dir, model, optimizer, image_datasets, arch="vgg13", hidden_units=[4096, 1000], dropouts=[0.5, 0.3]):
    checkpoint = {
        'hidden_units': hidden_units,
        'input_layer': 25088,
        'output_layer': 102,
        'dropouts': dropouts,
        'structure': f'models.{arch}(pretrained=True)',
        'epochs': 5,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets['train_data'].class_to_idx,
        'classifier': model.classifier}

    # Save the checkpoint
    torch.save(checkpoint, save_dir+f'/{arch}_checkpoint.pth')
    print(f"saved file at {save_dir}")

def load_checkpoint(filepath):
    """loads and rebulids a saved model"""
    #loading checkpoint from a file
    checkpoint = torch.load(filepath)

    # load the model and its specifications
    model = eval(checkpoint['structure'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def train_network(model, dataloaders, arch='vgg13', gpu=True, lr=0.01, epochs=1):
    # Use GPU if it's available
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
             device = torch.device("cpu")
             print("CUDA is not available!")
    else:
        device = torch.device("cpu")
    model.to(device);

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    ## define static variables
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train_loader']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                # evaluate the network using the valid dataloader
                loss, accuracy = evaluate_network(model, device, dataloaders['valid_loader'], criterion)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {loss/len(dataloaders['valid_loader']):.3f}.. "
                      f"Valid accuracy: {accuracy/len(dataloaders['valid_loader']):.3f}"
                     )
                running_loss = 0
                model.train()
                
    return optimizer # for save_checkpoint

def predict(image_path, model, category_names, gpu=True, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use GPU if it's available
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
             device = torch.device("cpu")
             print("CUDA is not available!")
    else:
        device = torch.device("cpu")
    model.to(device);
    # put the model in the evaluation mode
    model.eval()

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # reshape the processed image
    reshaped_img = process_image(image_path).unsqueeze_(dim=0).type(torch.FloatTensor).to(device)

    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        logps = model.forward(reshaped_img)

        # Calculate probabilities
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # get probabilities & names
    probs, class_names = mapping(model, top_p, top_class, category_names)
    
    return probs, class_names