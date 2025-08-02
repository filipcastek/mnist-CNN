import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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
    
model = CNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

image_path = r""#path k vasemu obrazku cislice, musi byt 28x28, cerne pozadi, bile cislo

img = Image.open(image_path).convert('L')

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)

predicted_class = predicted.item()
print(f"Predikovaná číslice je: {predicted_class}")

plt.imshow(img, cmap='gray')
plt.title(f"Predikovaná číslice: {predicted_class}")
plt.show()
