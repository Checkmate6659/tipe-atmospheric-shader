print("LOADING...")

from math import floor
from random import randrange
import numpy as np

import torch
from torch import nn

#why torch.accelerator not work
#device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")


NPOINTS = 64
#genfromtxt is a piece of crap, takes up LOADS of memory!
raw_data = np.zeros(NPOINTS**4 * 3)
with open("data.txt", "r") as fptr:
    string = fptr.read().split(", ")[:-1]
    for i, s in enumerate(string):
        raw_data[i] = float(s)

#reshape it into a N^4 array, with 3 channels
data = np.reshape(raw_data, (NPOINTS, NPOINTS, NPOINTS, NPOINTS, 3))
print("Done loading")


#Get non-uniformly picked random sample
def get_random_sample():
    ix = randrange(NPOINTS)
    iy = randrange(NPOINTS)
    iz = randrange(NPOINTS)
    iw = randrange(NPOINTS)

    #look more straight forward (not up or down)
    if randrange(3) > 0: #2/3 of the time
        #non-uniform sampling of z: more focus on the middle area where the highest brightnesses, and errors, occur
        raw = np.random.normal(0.5, 0.1) #tighter towards middle
        iz = np.clip(floor(raw * NPOINTS), 0, NPOINTS - 1)

    #spend more time in the evening part
    if randrange(3) > 0: #2/3 of the time
        #non-uniform sampling of y: more focus on the middle area where the highest brightnesses, and errors, occur
        raw = np.random.normal(0.5, 0.1) #tighter towards middle
        iy = np.clip(floor(raw * NPOINTS), 0, NPOINTS - 1)

    x = np.linspace(1e-6, 1, NPOINTS)[ix] #since infinite altitude poses some issues, this has been done in the data generation
    y = np.linspace(0, 1, NPOINTS)[iy]
    z = np.linspace(0, 1, NPOINTS)[iz]
    w = np.linspace(0, 1, NPOINTS)[iw]
    label = data[ix][iy][iz][iw]

    return x, y, z, w, label

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

#loss_fn = nn.L1Loss() #||.||_1 (kinda crap)
loss_fn = nn.MSELoss() #using ||.||_2 for error, to be just a bit closer to ||.||_inf (TODO: better?)
#loss_fn = LpLoss() #||.||_4
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)

print("Loss function: L2; Optimizer: SGD; lr: 5e-4")

loss_for_printing = nn.MSELoss()


#TODO: this! https://stackoverflow.com/questions/72650995/pytorch-what-is-the-proper-way-to-minimize-and-maximize-the-same-loss
#could this concentrate on particularly annoying samples or sth??


#Trains on 1 single sample, and returns loss
def train_iteration(model, loss_fn, optimizer):
    model.train() #train mode

    #create batch
    point = []
    result = []
    for _ in range(BATCH_SIZE):
        x, y, z, w, lbl = get_random_sample()
        point.append([x, y, z, w])
        result.append(lbl)

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
#TODO: test on whole dataset
def test(model, loss_fn):
    model.eval() #test mode

    total_loss = 0
    max_loss = 0
    with torch.no_grad():
        for _ in range(TEST_SIZE):
            x, y, z, w, result = get_random_sample()
            point = torch.as_tensor([x, y, z, w], dtype=torch.float32)
            result = torch.as_tensor(result, dtype=torch.float32)

            # Compute prediction error
            pred = model(point)
            cur_loss = loss_for_printing(pred, result).item()

            total_loss += cur_loss
            if cur_loss > max_loss:
                max_loss = cur_loss

    total_loss /= TEST_SIZE
    print(f"Average loss: {total_loss:>8f}")
    print(f"Max loss: {max_loss:>8f}\n")

#load checkpoint
#model.load_state_dict(torch.load("nets/chkpt.nn", weights_only=True))
#print("Starting from checkpoint")

epochs = 64
for t in range(epochs):
    print(f"================ EPOCH {t+1} ================\n")

    total_loss = 0
    for i in range(TRAIN_SIZE):
        total_loss += train_iteration(model, loss_fn, optimizer)
        if i % 32 == 31:
            total_loss /= 32
            print(f"Current avg loss: {total_loss:>8f}")
            total_loss = 0

    test(model, loss_fn)

    #save model!
    torch.save(model.state_dict(), f"nets/model_{t+1}.nn")

