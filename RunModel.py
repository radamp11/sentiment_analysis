import torch
import torch.nn as nn
import numpy as np
from Model import SentimentLSTM
from DataPreparation import obtain_data

'''loading data'''

batch_size = 50

data_path = "data/train/"
train_loader, valid_loader, test_loader, vocab_to_int = \
    obtain_data(data_path, seq_length=200, split_frac=0.8, batch_size=batch_size)

'''hyperparams:'''

vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

train_on_gpu = False
lr = 0.001

'''instantiate the model'''

net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, train_on_gpu=train_on_gpu)

'''carry on with the standard learning model'''

# loss and optimization functions

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params

epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip = 5  # gradient clipping

# move model to GPU, if available
if net.training_gpu_mode():
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                inputs = inputs.type(torch.LongTensor)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
