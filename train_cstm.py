import numpy as np
import pandas as pd
import torch


def def_net_test_samples(net, test_loader, sample = 16):
    train_features = next(iter(test_loader))

    # convert images to FloatTensors
    train_features = train_features.type(torch.FloatTensor)
    
    # forward pass to get net output
    net.eval()
    output_pts = net((train_features))

    return train_features, output_pts

    

def net_sample_output(net, test_loader):
    
    train_features, train_labels = next(iter(test_loader))

    # convert images to FloatTensors
    train_features = train_features.type(torch.FloatTensor)
    
    # forward pass to get net output
    net.eval()
    output_pts = net((train_features))
        
    # reshape to batch_size x 68 x 2 pts
    # output_pts = output_pts.view(output_pts.size()[0], 15, -1)
        
    return train_features, output_pts, train_labels

def train_model(model, trainloader, validloader, optimizer, scheduler, criterion, epochs = 5):
    history = {'val_loss': [], 'train_loss': []}

    for e in range(epochs):

        train_loss = 0.0
        model.train()
        for data, labels in trainloader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            data = data.type(torch.FloatTensor)
            target = model(data)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
        
        
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in validloader:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            # Forward Pass
            data = data.type(torch.FloatTensor)
            target = model(data)
            # Find the Loss
            loss = criterion(target,labels)
            # Calculate Loss
            valid_loss += loss.item()
    
        scheduler.step(valid_loss / len(validloader))

        history['val_loss'].append( valid_loss / len(validloader) )
        history['train_loss'].append( train_loss / len(trainloader) )
        print(f'Epoch {e+1} \
              \t\t Training Loss: {train_loss / len(trainloader)} \
              \t\t Validation Loss: {valid_loss / len(validloader)}')
    
    return history