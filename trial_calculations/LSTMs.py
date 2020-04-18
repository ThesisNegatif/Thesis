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
