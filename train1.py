# import torch
# from torch.utils.data import Dataset, random_split, DataLoader
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import random
# import os
# import matplotlib.pyplot as plt

_exp_name = "donkey train"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, random_split
from torchvision.datasets import DatasetFolder, VisionDataset, CIFAR10

# This is for the progress bar.
from tqdm.auto import tqdm
import random

if not os.path.isdir('./data'):
    os.mkdir('./data')

myseed = 881228  # Josh's birthday
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace = True)
])

# # However, it is also possible to use augmentation in the testing phase.
# # You may use train_tfm to produce a variety of images and then test using ensemble methods
# train_tfm = transforms.Compose([
#     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#     # You may add some transforms here.
#     transforms.RandomHorizontalFlip(p=0.7),
#     # ToTensor() should be the last one of the transforms.
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace = True)
# ])


if not os.path.isdir('./data'):
    os.mkdir('./data')

class DonkeyDataset(Dataset):
    def __init__(self,path,tfm=test_tfm,files = None):
        super(DonkeyDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),      # [128, 32, 32]

            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.cnn(x)
        #out = out.view(out.size()[0], -1)
        return self.fc(out)

batch_size = 500
_dataset_dir = f"./{_exp_name}"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
# train_set = DonkeyDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
# train_valid_set = CIFAR10(os.path.join(_dataset_dir,"training_validation"),train = True,transform = train_tfm,download = True)
# valid_set = DonkeyDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
data_set = DonkeyDataset(path = "./Data", tfm=test_tfm)
test_to_all_ratio = 0.2
valid_to_all_ratio = 0.2
train_to_all_ratio = 1 - test_to_all_ratio - valid_to_all_ratio
test_data_size = int(len(data_set)*test_to_all_ratio)
train_valid_data_size = len(data_set) - test_data_size
test_set, train_valid_set = random_split(data_set, [test_data_size, train_valid_data_size], generator=torch.Generator().manual_seed(myseed))
valid_data_size = int(len(train_valid_set)*valid_to_all_ratio/(valid_to_all_ratio + train_to_all_ratio))
train_data_size = train_valid_data_size - valid_data_size
valid_set, train_set = random_split(data_set, [valid_data_size, train_data_size], generator=torch.Generator().manual_seed(myseed))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# The number of training epochs and early_stop.
epochs = 40
early_stop = 300 # If no improvement in 'early_stop' epochs, early stop

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
print(model)
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5) 
#sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
plot_training_loss = []
plot_training_acc = []
plot_valid_loss = []
plot_valid_acc = []

for epoch in range(epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()
        sched.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        plot_training_loss.append(loss.item())
        plot_training_acc.append(acc.item())
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        plot_valid_loss.append(loss.item())
        plot_valid_acc.append(acc.item())
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > early_stop:
            print(f"No improvment {early_stop} consecutive epochs, early stopping")
            break

# df = pd.DataFrame()
# df["Training Loss"] = plot_training_loss
# df["Training Accuracy"] = plot_training_acc
# df.T.to_csv(f"{_exp_name}_training.csv",index = False)

# df = pd.DataFrame()
# df["Validation Loss"] = plot_valid_loss
# df["Validation Accuracy"] = plot_valid_acc
# df.T.to_csv(f"{_exp_name}_valid.csv",index = False)

# test_set = CIFAR10(os.path.join(_dataset_dir,"test"),train = False,transform = test_tfm,download = True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
labels_list = []
with torch.no_grad():
    for data,labels in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()
        labels_list += labels.tolist()
test_result = [prediction[i] == labels_list[i] for i in range(len(prediction))]
test_acc = sum(test_result)/len(test_result)
print(f"Accuracy on testing set is: {test_acc}")

#create test csv
# def pad4(i):
#     return "0"*(4-len(str(i)))+str(i)
# df = pd.DataFrame()
# df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
# df["Category"] = prediction
# df["Answer"] = labels_list
# df.to_csv(f"{_exp_name}.csv",index = False)
