import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from net import Net
from get_dataset import main as get_dataset

from config import PATH_TO_SAVE_NETWORK


def main():

    net = Net()
    trainset, testset = get_dataset()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # training
    print('Starting training...\n')
    EPOCHS = 3
    for epoch in range(EPOCHS):
        for data in tqdm(trainset):  # data is a batch of feature sets & labels
            X, y = data
            net.zero_grad()  # reset gradients
            output = net(X.view(-1, 28*28))  # -1: (batch-)size not known

            loss = F.nll_loss(output, y)
            loss.backward()  # magical
            optimizer.step()  # adjust weights

    # evaluation
    correct, total = 0, 0
    with torch.no_grad():  # don't train on out_of_sample data
        for data in testset:
            X, y = data
            output = net(X.view(-1, 28*28))
            for idx, i in enumerate(output):
                if np.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print('Accuracy:', round(correct / total, 3))  # ~0.97, great accuracy!
    torch.save(
        net, os.path.join(PATH_TO_SAVE_NETWORK, 'handwritten_digit_classifier')
    )


if __name__ == "__main__":
    main()
