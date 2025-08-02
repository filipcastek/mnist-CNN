import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.ion()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()


loss_values = []
accuracy_values = []
steps = []

loss_line, = ax1.plot([], [], label='Ztráta', color='red')
ax1.set_xlabel('Kroky')
ax1.set_ylabel('Ztráta')
ax1.set_title('Ztráta během tréninku')
ax1.grid(True)
ax1.legend()

acc_line, = ax2.plot([], [], label='Přesnost', color='blue')
ax2.set_xlabel('Kroky')
ax2.set_ylabel('Přesnost')
ax2.set_title('Přesnost během tréninku')
ax2.grid(True)
ax2.legend()


def update_plots(step, loss, accuracy):
    loss_values.append(loss)
    accuracy_values.append(accuracy)
    steps.append(step)

    loss_line.set_data(steps, loss_values)
    ax1.relim()
    ax1.autoscale_view()
    fig1.canvas.draw()
    fig1.canvas.flush_events()

    acc_line.set_data(steps, accuracy_values)
    ax2.relim()
    ax2.autoscale_view()
    fig2.canvas.draw()
    fig2.canvas.flush_events()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Zařízení:", device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, kernel_size=5, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2)
        )#diamond shaped NN
        self.fc_stack = nn.Sequential(
            nn.Flatten(), #  přidáno pro zploštění výstupu konvoluční vrstvy
            nn.Linear(256 * 6 * 6, 1024),  # ← upraveno z 7*7 na NE 5x5 proto to nejde pac 6x6 kvůli změnám velikosti po konvolucích achh proto to neslo
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256), 
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10) # 10 tříd pro klasifikaci, gelu pro fine detailing 
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100 * correct / total



epochs = 12
step_count = 0
best_val_accuracy = 0.0
epochs_without_improvement = 0
early_stopping_patience = 5

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        if epoch < 6:
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        optimizer.zero_grad()
        outputs = model(images).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        step_count += 1
        if i % 100 == 99:
            avg_loss = running_loss / 100
            accuracy = 100 * correct / total
            print(f"[{epoch+1}, {i+1}] ztráta(avg): {avg_loss:.4f}, přesnost: {accuracy:.2f}%")
            update_plots(step_count, avg_loss, accuracy)
            running_loss = 0.0
            correct = 0
            total = 0

    val_accuracy = evaluate(model, testloader, device)
    print(f"validace po epoše {epoch+1}: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Uloženo jako nejlepší model.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping: žádné zlepšení přes", early_stopping_patience, "epoch.")
            break


torch.save(model.state_dict(), 'cnn_mnist.pth')
print("CNN trénink dokončen a model uložen.")


plt.ioff()
plt.show()