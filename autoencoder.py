# importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os

# -------------- Paths -----------
BASE = os.getcwd()
DATASET_PATH_MOVIE = BASE + '/Dataset/ml-1m/movies.dat'
DATASET_PATH_USERS = BASE + '/Dataset/ml-1m/users.dat'
DATASET_PATH_RATING = BASE + '/Dataset/ml-1m/ratings.dat'
TRAINING_SET_DIR = BASE + '/Dataset/ml-100k/u1.base'
TEST_SET_DIR = BASE + '/Dataset/ml-100k/u1.test'
MODEL_DIR = BASE + '/Model/'

# importing the dataset

movies = pd.read_csv(DATASET_PATH_MOVIE
                     , sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(DATASET_PATH_USERS, sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(DATASET_PATH_RATING, sep='::', header=None, engine='python', encoding='latin-1')

# preparing the training set and test set
training_set = pd.read_csv(TRAINING_SET_DIR, delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv(TEST_SET_DIR, delimiter='\t')
test_set = np.array(test_set)

# number of user and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# restructuring the data

def convert(data):
    new_list = []
    for user_id in range(1, nb_users + 1):
        user_movies = data[:, 1][data[:, 0] == user_id]
        user_ratings = data[:, 2][data[:, 0] == user_id]
        rating = np.zeros(nb_movies)
        rating[user_movies - 1] = user_ratings
        new_list.append(list(rating))
    return new_list


training_set = convert(training_set)
test_set = convert(test_set)

# converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# creating the architecture of the neural network

class StackedAutoEncoders(nn.Module):
    def __init__(self, ):
        super(StackedAutoEncoders, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


encoder = StackedAutoEncoders()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(encoder.parameters(), lr=0.01, weight_decay=0.5)

# generating the path to save the model

try:
    os.mkdir(MODEL_DIR)
    print("Directory Created Successfully")
except FileExistsError:
    pass

# training the encoder
no_epoch = 200
for epoch in range(1, no_epoch + 1):
    train_loss = 0
    counter = 0.
    for users in range(nb_users):
        input_data = Variable(training_set[users]).unsqueeze(0)
        target_data = input_data.clone()
        if torch.sum(target_data.data > 0) > 0:
            output = encoder(input_data)
            target_data.require_grad = False
            output[target_data == 0] = 0
            loss = criterion(output, target_data)
            mean_corrector = nb_movies / float(torch.sum(target_data.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            counter += 1.
            optimizer.step()
    print("Epoch:" + str(epoch) + 'Train Loss:' + str(train_loss / counter))
#     saving the trained Model To The Directory
torch.save(encoder, MODEL_DIR + 'myModel')
# testing the encoders
test_loss = 0
counter = 0.
for users in range(nb_users):
    test_input_data = Variable(training_set[users]).unsqueeze(0)
    test_target_data = Variable(test_set[users]).unsqueeze(0)
    if torch.sum(test_target_data.data > 0) > 0:
        output = encoder(test_input_data)
        test_target_data.require_grad = False
        output[test_target_data == 0] = 0
        loss = criterion(output, test_target_data)
        mean_corrector = nb_movies / float(torch.sum(test_target_data.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        counter += 1.
print('Test Loss:' + str(test_loss / counter))

# ==================================================================================================================================================================
