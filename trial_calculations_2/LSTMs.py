class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.

        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size. embedding_dim
            lstm_size : The LSTM layer size. hidden_dim
            output_size : The output size.
            lstm_layers : The number of LSTM layers. n_layers
            dropout : The dropout probability.
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)


        # Setup additional layers
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers,
                            dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)



    def init_hidden(self, batch_size):
        """
        Initializes hidden state

        Parameters
        ----------
            batch_size : The size of batches.

        Returns
        -------
            hidden_state

        """

        # TODO Implement

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        #hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        #             weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                     weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

        return hidden


    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.

        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """

        # TODO Implement
        nn_input = nn_input.long()
        #batch_size = nn_input.size(1)
        #print('input size', nn_input.size())


        embedding = self.embedding(nn_input)

        r_output, hidden_state = self.lstm(embedding, hidden_state)
        #print('lstm size', r_output)

        '''
        r_output = r_output.contiguous().view(-1, self.lstm_size)
        #print('lstm size reshaped',  r_output.size())
        out = self.dropout(r_output)

        out = self.fc(out)
        logps = self.log_softmax(out)
        #print('softmax size', logps.size())
        logps = logps.view(batch_size, -1)
        logps = logps[:, -1]
        #print('softmax reshaped', logps.size())
        '''

        r_output = r_output[-1,:,:]
        #print('lstm size reshaped',  r_output.size())
        out = self.dropout(r_output)
        out = self.fc(out)
        logps = self.log_softmax(out)
        #print('softmax size', logps.size())


        return logps, hidden_state

#view the model
"""
model_lstm = TextClassifier(len(vocab), 10, 6, 5, dropout=0.1, lstm_layers=2)
model_lstm.embedding.weight.data.uniform_(-1, 1)
input = torch.randint(0, 1000, (5, 4), dtype=torch.int64)
#print('model input', input)
hidden = model_lstm.init_hidden(4)

logps, _ = model_lstm.forward(input, hidden)
print(logps)
"""

def dataloader(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    """
    Build a dataloader.
    """
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages) #total number of twits

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]

        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            # Left pad!
            start_idx = max(sequence_length - len(token_tensor), 0) #returns 0 is len(token_tensor) > seqeuence_length
            batch[start_idx:, batch_num] = token_tensor[:sequence_length] #replace each row in batch with the token

        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])

        yield batch, label_tensor

val_idx = int(len(token_ids)*0.8)

train_features = token_ids[:val_idx]
valid_features = token_ids[val_idx:]
train_labels = sentiments[:val_idx]
valid_labels = sentiments[val_idx:]

if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_lstm = TextClassifier(len(vocab)+1, 1024, 512, 3, lstm_layers=2, dropout=0.2)
model_lstm = TextClassifier(len(vocab)+1, 1024, 512, 3, lstm_layers=2, dropout=0.1)
model_lstm.embedding.weight.data.uniform_(-1, 1)
model_lstm.to(device)
#(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
########################

#TRAIN the network:
epochs = 4
batch_size = 20
learning_rate = 0.001
clip=5
#sequence_length = 20
sequence_length = 30

print_every = 100
criterion = nn.NLLLoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=learning_rate)
model_lstm.train()

for epoch in range(epochs):
    print('Starting epoch {}'.format(epoch + 1))

    steps = 0
    #hidden = model_lstm.init_hidden(batch_size)

    for text_batch, labels in dataloader(
            train_features, train_labels, batch_size=batch_size, sequence_length=sequence_length, shuffle=True):
        #print('Starting batch step {}'.format(steps))
        steps += 1
        hidden = model_lstm.init_hidden(labels.shape[0])

        # Set Device
        text_batch, labels = text_batch.to(device), labels.to(device)
        for each in hidden:
            each.to(device)
        #hidden = tuple([each.data for each in hidden])
        model_lstm.zero_grad()

        # TODO Implement: Train Model
        output, hidden = model_lstm(text_batch, hidden)
        loss = criterion(output, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), clip)
        optimizer.step()

        if steps % print_every == 0:
            model_lstm.eval()
            #val_h = model_lstm.init_hidden(batch_size)
            val_losses = []
            accuracy = []

            with torch.no_grad():
                for text_batch, labels in dataloader(valid_features, valid_labels, batch_size=batch_size, sequence_length=sequence_length, shuffle=True):
                    text_batch, labels = text_batch.to(device), labels.to(device)
                    val_h = model_lstm.init_hidden(labels.shape[0])
                    for each in val_h:
                        each.to(device)
                    #val_h = tuple([each.data for each in val_h])

                    output, val_h = model_lstm(text_batch, val_h)
                    val_loss = criterion(output, labels)
                    val_losses.append(val_loss.item())

                    probabilities = torch.exp(output)
                    top_probability, top_class = probabilities.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())



            print("Epoch: {}/{}...".format(epoch+1, epochs),
                 "Step: {}...".format(steps),
                 "Training Loss: {:.6f}...".format(loss.item()),
                 "Val loss: {:.6f}".format(np.mean(val_losses)),
                 "Val accuracy: {:.6f}".format(np.mean(accuracy)))

            model_lstm.train()


#######################################################
#######################################################













class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.

        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size. embedding_dim
            lstm_size : The LSTM layer size. hidden_dim
            output_size : The output size.
            lstm_layers : The number of LSTM layers. n_layers
            dropout : The dropout probability.
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)


        # Setup additional layers
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers,
                            dropout=dropout, batch_first=False, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_size*2, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)



    def init_hidden(self, batch_size):
        """
        Initializes hidden state

        Parameters
        ----------
            batch_size : The size of batches.

        Returns
        -------
            hidden_state

        """

        # TODO Implement

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        #hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        #             weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        hidden = (weight.new(self.lstm_layers*2, batch_size, self.lstm_size).zero_(),
                     weight.new(self.lstm_layers*2, batch_size, self.lstm_size).zero_())

        return hidden


    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.

        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """

        nn_input = nn_input.long()
        #batch_size = nn_input.size(1)
        #print('input size', nn_input.size())


        embedding = self.embedding(nn_input)

        r_output, hidden_state = self.lstm(embedding, hidden_state)
        #print('lstm size', r_output)

        r_output = r_output[-1,:,:]
        #print('lstm size reshaped',  r_output.size())
        out = self.dropout(r_output)
        out = self.fc(out)
        logps = self.log_softmax(out)
        #print('softmax size', logps.size())


        return logps, hidden_state
