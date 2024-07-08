import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if GPU is available and set the device
device = torch.device('cuda')
print(device)


# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Hyperparameters
input_size = 784  # 28x28 image size
hidden_size = 128
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 50

# Load the MNIST dataset and create data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer and move them to the device
model = MLP(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

result_file = open('Results.txt', 'w')

# Training loop
for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for images, labels in train_loader:
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    accuracy = 100 * total_correct / total_samples
    print(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%')
    result_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Accuracy: {accuracy:.2f}%\n')

# Test the model on the test set
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * total_correct / total_samples
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    result_file.write(f'Test Accuracy: {test_accuracy:.2f}%\n')

result_file.close()