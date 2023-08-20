from model import BinaryClassifierWithResidual
from dataloader import load_training, load_testing, get_dataset_detail
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

EPOCH = 100
BATCH_SIZE = 8
TRAIN_DATASET_LENGTH = get_dataset_detail()["train_len"]
TEST_DATASET_LENGTH = get_dataset_detail()["test_len"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BinaryClassifierWithResidual().to(device)
# model.load_state_dict(torch.load("models_views/3.pth"))
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
loss_fn = nn.BCELoss()


def test_model(model, batch_size):
    Y = []
    Y_cap = []
    for idx in range(0, TEST_DATASET_LENGTH // batch_size):
        x, y = load_testing(idx, BATCH_SIZE)
        if x is None:
            break
        x = torch.tensor(x, dtype=torch.float).to(device)
        y_cap = model(x)
        y_cap = np.array([0 if a < 0.5 else 1 for a in y_cap.detach().cpu().numpy()])
        Y.extend(y)
        Y_cap.extend(y_cap)
    Y = np.array(Y)
    Y_cap = np.array(Y_cap)[:, np.newaxis]
    return np.sum((Y == Y_cap).astype(int)) / Y.shape[0]


for e in range(EPOCH):
    progress_bar = tqdm(
        total=TRAIN_DATASET_LENGTH // BATCH_SIZE,
        desc="Processing epoch " + str(e),
        unit="iteration",
    )
    for idx in range(0, TRAIN_DATASET_LENGTH // BATCH_SIZE):
        X, Y = load_training(idx, BATCH_SIZE)
        if X is None:
            break
        X, Y = (
            torch.tensor(X, dtype=torch.float).to(device),
            torch.tensor(Y, dtype=torch.float).to(device),
        )

        optimizer.zero_grad()

        Y_cap = model(X)
        loss = loss_fn(Y_cap, Y)

        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    progress_bar.close()
    print(e, test_model(model, 16))
    torch.save(model.state_dict(), "models_comments/" + str(e) + ".pth")
