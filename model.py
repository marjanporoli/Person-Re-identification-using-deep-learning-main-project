import torch
import torch.nn as nn
import timm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage import io
import pandas as pd

DATA_DIR = r"C:/Users/USER/Desktop/reid_frontend/Market-1501-v15.09.15/bounding_box_train/"
CSV_FILE = "C:/Users/USER/Desktop/reid_frontend/Person-Re-Id-Dataset/train.csv"

BATCH_SIZE = 32
LR = 0.001
EPOCHS = 15

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
df = pd.read_csv(CSV_FILE)
train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)

class APN_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        A_img = io.imread(DATA_DIR + row.Anchor)
        P_img = io.imread(DATA_DIR + row.Positive)
        N_img = io.imread(DATA_DIR + row.Negative)

        A_img = torch.from_numpy(A_img).permute(2, 0, 1) / 255.0
        P_img = torch.from_numpy(P_img).permute(2, 0, 1) / 255.0
        N_img = torch.from_numpy(N_img).permute(2, 0, 1) / 255.0

        return A_img, P_img, N_img

trainset = APN_Dataset(train_df)
validset = APN_Dataset(valid_df)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

class APN_Model(nn.Module):
    def __init__(self, emb_size=512):
        super(APN_Model, self).__init__()

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features,
                                                 out_features=emb_size)

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings

model = APN_Model().to(DEVICE)

# Train and evaluate functions

def train_fn(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for A, P, N in tqdm(dataloader):
        A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)

        A_embs = model(A)
        P_embs = model(P)
        N_embs = model(N)

        loss = criterion(A_embs, P_embs, N_embs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(trainset)

def eval_fn(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for A, P, N in tqdm(dataloader):
            A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)

            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)

            loss = criterion(A_embs, P_embs, N_embs)

            total_loss += loss.item()

    return total_loss / len(validset)

# Define loss function and optimizer
criterion = nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
#best_valid_loss = np.Inf

#for epoch in range(EPOCHS):
    #train_loss = train_fn(model, trainloader, optimizer, criterion)
    #valid_loss = eval_fn(model, validloader, criterion)
   # print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")

# Save the PyTorch model
torch.save(model.state_dict(), 'reidentification.pth')
