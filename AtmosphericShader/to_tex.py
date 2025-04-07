import numpy as np
from PIL import Image

NPOINTS = 2 #64
#fname = "data%dc" % NPOINTS
fname = "test"
print("Loading data...")
#genfromtxt is a piece of crap
raw_data = np.zeros(NPOINTS**4 * 3)
with open(fname + ".txt", "r") as fptr:
    string = fptr.read().split(", ")[:-1]
    for i, s in enumerate(string):
        raw_data[i] = float(s)
print("Done loading data")
print(raw_data)

#reshape it into a N^2 * N^2 image
#x coord will be param1 * NPOINTS + param2
#y coord will be param3 * NPOINTS + param4
print("Reshaping...")
raw_data = np.reshape(raw_data, (NPOINTS*NPOINTS, NPOINTS*NPOINTS, 3))
print("Done reshaping")

#clamp to be positive, and divide out EACH COLUMN SEPARATELY to be 0 to 255, but not quite 256
print("Normalizing...")
raw_data = np.maximum(raw_data, 0) #all positive!
red_max = np.max(raw_data[:,:,0]) #calculate red, green and blue max brightnesses
green_max = np.max(raw_data[:,:,1])
blue_max = np.max(raw_data[:,:,2])
print("RGB MAX:", red_max, green_max, blue_max) #PRINT THEM! Very important!
#didn't find another way to rescale each column
#TODO: optimize! This is really slow! vectorization would *probably* help
data = np.array([[(r/red_max, g/green_max, b/blue_max) for (r, g, b) in row] for row in raw_data])
data *= 256 - 1e-9 #go up to 256 (almost)
print("Done normalizing")

#save it
print("Saving image...")
im = Image.fromarray(data.astype(np.uint8), mode="RGB")
im.save(fname + ".png")
print("Saving done")
