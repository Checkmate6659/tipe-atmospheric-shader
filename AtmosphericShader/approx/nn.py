print("LOADING LIBS...")

from math import floor
from random import randrange
import numpy as np
import subprocess

import torch
from torch import nn

#why torch.accelerator not work
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")


#generate samples OFF GRID! no more DUMB overfitting!
def generate_samples(n):
    sampler = subprocess.Popen(["sampler.exe", str(n)]) #call C program to generate n samples
    sampler.communicate() #wait for it to finish

    #format: each element is ([x, y, z, w], np.array([red, green, blue]))
    samples = []

    with open("samples.txt", "r") as fptr:
        lines = fptr.read().split("\n")[:-1]
        for line in lines:
            sx, sy, sz, sw, sr, sg, sb = line.split(",")
            samples.append(([float(sx), float(sy), float(sz), float(sw)], np.array((float(sr), float(sg), float(sb)))))

    #print(*samples[:100], sep="\n")

    return samples

#Model definition
#Conventional model, 4->16->16->3, used beforehand
#trying to use L1, as well as slightly bigger nn, to increase significance of red and green components in loss
#Model definition
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            #nn.ReLU(),
            nn.Tanh(),
            nn.Linear(16, 16),
            #nn.ReLU(),
            nn.LeakyReLU(0.0625),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

model = Model().to(device)
print(model)


#NOTE: too big numbers may result in NaN!!! i think thats the issue, not autograd
#also, converting a list of ndarrays to a tensor is really slow, fix it mb?
#also mb try to do max(4th power * coef and clamped, 2nd power) so that it actually does sth when close to minimum?
#differentiability issue tho...
class LpLoss(nn.Module):
    def __init__(self):
        super(LpLoss, self).__init__()
    def forward(self, predictions, targets):
        #delta = torch.clamp(predictions - targets, min=-1e8, max=1e8) #basically lower than (2^127)^(1/4)
        delta = torch.clamp((predictions - targets) / 100, min=-1e8, max=1e8) #to not go absolutely NUTS

        delta2 = torch.square(delta) #regular multiply is not differentiable for some reason...
        delta4 = torch.square(delta2)
        #delta8 = torch.square(delta4)

        lp_loss = torch.mean(delta4) #squaring once works, squaring more doesn't??
        #print(lp_loss)
        return lp_loss


#Minimization hyperparameters
TRAIN_SIZE = 512
TEST_SIZE = 32768 #16384
BATCH_SIZE = 512
print("TRAIN_SIZE =", TRAIN_SIZE)
print("TEST_SIZE =", TEST_SIZE)
print("BATCH_SIZE =", BATCH_SIZE)

loss_fn = nn.L1Loss() #||.||_1 (to fail less on colors that are not blue...)
#loss_fn = nn.MSELoss() #using ||.||_2 for error, to be just a bit closer to ||.||_inf (TODO: better?)
#loss_fn = LpLoss() #||.||_4

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4) #NANS with unconventional models
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6) #weight decay: regularization (normal: 1e-5)

print("Loss function: L1; Optimizer: Adam; lr: 1e-4, weight_decay: 1e-6")


#TODO: this! https://stackoverflow.com/questions/72650995/pytorch-what-is-the-proper-way-to-minimize-and-maximize-the-same-loss
#could this concentrate on particularly annoying samples or sth??


#Trains on 1 single sample, and returns loss
def train_iteration(model, loss_fn, optimizer, samples):
    model.train() #train mode

    #create batch
    point = [p for (p, r) in samples]
    result = [r for (p, r) in samples]

    point = torch.as_tensor(point, dtype=torch.float32)  #float32 because weights are float32
    result = torch.as_tensor(np.array(result), dtype=torch.float32)
    #print(point)
    #print(result)

    # Compute prediction error
    pred = model(point)
    loss = loss_fn(pred, result)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

#Test over lots of random samples
#NOTE: it only does max loss on THOSE samples! not any potential outliers!
def test(model, loss_fn):
    model.eval() #test mode

    samples = generate_samples(TEST_SIZE)

    total_loss = 0
    max_loss = 0
    with torch.no_grad():
        for i in range(TEST_SIZE):
            point, result = samples[i]
            point = torch.as_tensor(point, dtype=torch.float32)
            result = torch.as_tensor(result, dtype=torch.float32)

            # Compute prediction error
            pred = model(torch.stack((point,))) #should fix model for this ig...
            cur_loss = loss_fn(pred, result).item()

            total_loss += cur_loss
            if cur_loss > max_loss:
                max_loss = cur_loss

    total_loss /= TEST_SIZE
    print(f"Average loss: {total_loss:>8f}")
    print(f"Max loss: {max_loss:>8f}\n")

#load checkpoint
model.load_state_dict(torch.load("nets/chkpt.nn", weights_only=True))
print("Starting from checkpoint")

iterations = 64
for t in range(iterations):
    print(f"================ ITERATION {t+1} ================\n")

    total_loss = 0
    samples = generate_samples(BATCH_SIZE * TRAIN_SIZE)
    for i in range(TRAIN_SIZE):
        batch_samples = samples[:BATCH_SIZE] #separate relevant samples
        samples = samples[BATCH_SIZE:]

        #train and add up losses
        total_loss += train_iteration(model, loss_fn, optimizer, batch_samples)
        if i % 128 == 127:
            total_loss /= 128
            print(f"Current avg loss: {total_loss:>8f}")
            total_loss = 0

    test(model, loss_fn)

    #save model!
    torch.save(model.state_dict(), f"nets/model_{t+1}.nn")

