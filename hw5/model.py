import numpy as np
import torch
import torch.nn as nn

class Custom_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Parameters:
        Inputting the input_size and hidden_size. For our purposes, these are the same(embedding size)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #initialize parameters
        self.W_f = nn.Linear(input_size + hidden_size, input_size, bias=False)
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_i = nn.Linear(input_size + hidden_size, input_size, bias=False)
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_C = nn.Linear(input_size + hidden_size, input_size, bias=False)
        self.b_C = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Linear(input_size + hidden_size, input_size, bias=False)
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    #The following gate functions come from Chris Olah's blog linked in the homework

    def f_t(self, h_tmin1, x_t):
        product = self.W_f(torch.cat([h_tmin1, x_t]))
        sum = product + self.b_f

        return self.sig(sum)
    
    def i_t(self, h_tmin1, x_t):
        product = self.W_i(torch.cat([h_tmin1, x_t]))
        sum = product + self.b_i
        return self.sig(sum)


    def Ctilde_t(self, h_tmin1, x_t):
        product = self.W_C(torch.cat([h_tmin1, x_t]))
        sum = product + self.b_C
        return self.tanh(sum)
    
    def C_t(self, h_tmin1, x_t, c_tmin1):
        ft = self.f_t(h_tmin1, x_t)
        it = self.i_t(h_tmin1, x_t)
        Ctildet = self.Ctilde_t(h_tmin1, c_tmin1)
        return ft*c_tmin1 + it*Ctildet

    def o_t(self, h_tmin1, x_t):
        product = self.W_o(torch.cat([h_tmin1, x_t]))
        sum = product + self.b_o
        return self.sig(sum)

    def h_t(self, h_tmin1, x_t, c_tmin1):
        ot = self.o_t(h_tmin1, x_t)
        return ot * self.tanh(self.C_t(h_tmin1, x_t, c_tmin1))
        
    def output(self, h_tmin1, c_tmin1, x_t):
        """
        outputs hidden state, then cell state
        """
        return self.h_t(h_tmin1, x_t, c_tmin1), self.C_t(h_tmin1, x_t, c_tmin1)    
        

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cust_lstm = Custom_LSTM(input_size, hidden_size)

    def forward(self, embedding_seq):
        ht = torch.zeros(self.hidden_size)
        ct = torch.zeros(self.hidden_size)

        for emb_word in embedding_seq:
            ht, ct = self.cust_lstm.output(ht, ct, emb_word)

        return ht, ct
    

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_embed):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cust_lstm = Custom_LSTM(input_size, hidden_size)
        self.vocab_embed = vocab_embed


    def round_to_word(self, prediction):
        """
        all_embeddings is a list (with length= length of vocab) of all possible embeddings
        We'd like for each prediction to be an actual word (even if it's the wrong one)

        Each entry in the vocabulary is a candidate, if the norm of the prediction minus
        the candidate embedding is less than the current, then the current embedding becomes
        the new candidate
        """
        index = 20
        best_guess = self.vocab_embed[index]
        best_dist = torch.inf

        for embedded in self.vocab_embed:
            candidate_dist = torch.linalg.norm(prediction - embedded)
            if candidate_dist < best_dist:
                best_guess = embedded
                best_dist = candidate_dist
            
        return best_guess
            

    def forward(self, target_seq, encoder_h, encoder_c, tf):
        ht = encoder_h
        ct = encoder_c
        predictions = []
        for i, emb_word in enumerate(target_seq):
            ht, ct = self.cust_lstm.output(ht, ct, emb_word)
            #ht = self.round_to_word(ht)
            rng = np.random.rand()
            tf_ratio = 0.0
            if tf and rng < tf_ratio:
                    ht = emb_word
            predictions.append(ht)

        predictions = torch.stack(predictions)
        predictions.requires_grad_()
        return predictions


class S2S(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_sequence, target_sequence, tf=False):
        h, c = self.encoder(input_sequence)
        prediction = self.decoder(target_sequence, h, c, tf)
        return prediction

        


