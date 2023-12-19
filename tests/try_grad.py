import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)

        # Fully connected layer
        self.fc = nn.Linear(32 * 32 * 32, 10)  # Assuming input image size of 32x32

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc(x)

        return x


if __name__ == "__main__":
    # Define the transformations to apply to the CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Create the CIFAR-10 train dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="~/Data/cifar10", train=False, download=True, transform=transform
    )

    # Create the CIFAR-10 train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )

    # Create an instance of the model
    model = SimpleModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Variable to store the sum of gradients
    grad_sum = []
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}")
        for i, (images, labels) in enumerate(train_loader):
            tmp_grad_sum = 0.0
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the sum of gradients
            for param in model.parameters():
                if param.grad is not None:
                    tmp_grad_sum += param.grad.abs().sum().item()

            grad_sum.append(tmp_grad_sum)

            # Optimize
            optimizer.step()

            # Print the sum of gradients in each iteration
            print(f"Iteration: {i+1}, Gradient Sum: {tmp_grad_sum}")
