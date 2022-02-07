import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import layers
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

in_size=416
#stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
stats = (0.5), (0.5)
def build_path(input_dir):
    dataset = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for x in filenames:
            if x.endswith(".jpg"):
                dataset.append(os.path.join(dirpath, x))
    return dataset
           
def load_image_binary(img,input_size = in_size):
    image =Image.open(img).convert('RGB')
    #image =Image.open(img).convert('L')
    image = transforms.Resize((in_size, in_size))(image)
    #print('Loaded image...')
    #print('Image Size: {}'.format(image.size))
    return image

def prepare_dataset(dir_path):
    arr=[]
    dataset=build_path(dir_path)
    for i in dataset:
        arr.append(load_image_binary(i,in_size))
    return arr


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#training function
def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        #data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            #data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images

def save_reconstructed_images(recon_images, epoch):
    save_image(recon_images.cpu(), f"{epoch}.jpg")

class testkitDataset(Dataset):
    def __init__(self, X):
        'Initialization'
        self.X = X
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        return X
    transform = transforms.Compose([
        #T.ToPILImage(),
        #T.Resize(image_size),
        transforms.ToTensor(),
        #transforms.Normalize(*stats)
        ])
    '''
    transform = transforms.Compose([
        #T.ToPILImage(),
        #T.Resize(image_size),
        transforms.ToTensor()
        #transforms.Normalize(*stats)
        ])
        '''
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dir=r'./data/project/dataset/full/mixed_dataset'
test_dir=r'./data/project/dataset/tmp_val/val_test_kit'
anomaly_dir='./data/project/dataset/other-20211216T024534Z-001/other'
train_dataset =prepare_dataset(train_dir)
test_dataset=prepare_dataset(test_dir)
anomaly_dataset = prepare_dataset(anomaly_dir)
transformed_dataset = testkitDataset(train_dataset)
test_transformed_dataset=testkitDataset(test_dataset)
anomaly_transformed_dataset=testkitDataset(anomaly_dataset)
train_dl = DataLoader(transformed_dataset, batch_size=32, shuffle=True)
test_dl= DataLoader(test_transformed_dataset,batch_size=32)
anomaly_dl=DataLoader(anomaly_transformed_dataset,batch_size=32)

"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=256*13*13, zDim=1024):
        super(VAE, self).__init__()
        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 32,3,stride=4,padding=1)
        self.encConv2 = nn.Conv2d(32, 64,3,stride=2,padding=1)
        #self.encPool1 = nn.MaxPool2d(4, 4)
        self.encConv3 = nn.Conv2d(64, 128,3,stride=2,padding=1)
        self.encConv4 = nn.Conv2d(128, 256,3,stride=2,padding=1)
        #self.encPool2 = nn.MaxPool2d(2, 2)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(256, 128,2,stride=2,padding=0)
        self.decConv2 = nn.ConvTranspose2d(128, 64,2,stride=2,padding=0)
        self.decConv3 = nn.ConvTranspose2d(64, 32,2,stride=2,padding=0)
        self.decConv4 = nn.ConvTranspose2d(32, imgChannels,4,stride=4,padding=0)
        
        
    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        #x = self.encPool1
        x = F.relu(self.encConv2(x))
        #x = self.encPool1
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        #x = self.encPool2
        x = x.view(-1, 256*13*13)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 256, 13, 13)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = torch.sigmoid(self.decConv4(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar
      
    
"""
Initialize Hyperparameters
"""
bs=256
learning_rate = 0.0001
num_epochs = 1000
grid_images = []#to save reconstructed image
criterion = nn.BCELoss(reduction='sum')


"""
Initialize the network and the Adam optimizer
"""
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss=[]
valid_loss=[]
'''
Training the network for a given number of epochs
The loss after every epoch is printed
'''

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} of {num_epochs}")
    train_epoch_loss = train(
        model, train_dl, transformed_dataset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, test_dl, test_transformed_dataset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    #save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    #image_grid = make_grid(recon_images.detach().cpu())
    #grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

torch.save(model, r'./data/project/variationalAE_RGB_v8.pt')