# SalientSleepNet模型的PyTorch代码框架

import torch
import torch.nn as nn

class SalientSleepNet(nn.Module):
    def __init__(self, num_modalities):
        super(SalientSleepNet, self).__init__()

        # Multi-scale CNN for saliency map generation
        self.ms_cnn = MS_CNN(num_modalities)

        # Fusion network for sleep stage classification
        self.fusion_net = FusionNet(num_modalities)

    def forward(self, x):
        # Generate multi-modal saliency maps
        saliency_maps = self.ms_cnn(x)

        # Fuse multi-modal saliency maps and predict sleep stages
        sleep_stages = self.fusion_net(saliency_maps)

        return sleep_stages

class MS_CNN(nn.Module):
    def __init__(self, num_modalities):
        super(MS_CNN, self).__init__()

        # Define multi-scale convolutional layers for each modality
        self.conv1 = nn.Conv2d(num_modalities, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Define pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Generate multi-scale feature maps for each modality
        feat_maps = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            out = torch.relu(self.conv1(x_i))
            out = self.pool(out)
            out = torch.relu(self.conv2(out))
            out = self.pool(out)
            out = torch.relu(self.conv3(out))
            feat_maps.append(out)

        # Concatenate feature maps from each modality
        feat_maps = torch.cat(feat_maps, dim=1)

        # Generate saliency map using convolutional layers
        saliency_map = self.conv3(feat_maps)

        return saliency_map

class FusionNet(nn.Module):
    def __init__(self, num_modalities):
        super(FusionNet, self).__init__()

        # Define convolutional layers for fusion network
        self.conv1 = nn.Conv2d(num_modalities, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Define pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers for sleep stage classification
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 5) # Five possible sleep stages

    def forward(self, x):
        # Generate multi-modal saliency maps
        saliency_maps = torch.split(x, 1, dim=1)

        # Generate feature maps for each saliency map and concatenate them
        feat_maps = []
        for saliency_map in saliency_maps:
            out = torch.relu(self.conv1(saliency_map))
            out = self.pool(out)
            out = torch.relu(self.conv2(out))
            out = self.pool(out)
            out = torch.relu(self.conv3(out))
            out = out.view(-1, 64 * 6 * 6) 
            feat_maps.append(out)
        feat_maps = torch.cat(feat_maps, dim=1)

        # Classify sleep stages using fully connected layers
        out = torch.relu(self.fc1(feat_maps))
        out = self.fc2(out)

        return out


import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Convert inputs and labels to tensors
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Evaluate the model on test set
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        # Convert inputs and labels to tensors
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Compute accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %d %%' % (
        100 * correct / total))
