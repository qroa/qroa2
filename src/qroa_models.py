import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformers import AutoModel
import numpy as np
import random
from nltk.corpus import words


    
class SurrogateModel(nn.Module):
    """
    A surrogate model for embedding-based optimization tasks.

    This model is designed to work with a set of precomputed embeddings (ref_emb) and predicts an output value based on a combination of these embeddings. The model consists of a series of fully connected layers and batch normalization layers.

    Attributes:
    emb_dim (int): Dimension of the embeddings.
    len_coordinates (int): The number of coordinates (or embeddings) used in each input.
    emb (torch.Tensor): A tensor containing the reference embeddings.
    fc1, fc2, fc3 (nn.Linear): Fully connected layers of the model.

    Methods:
    forward(str_id): Performs a forward pass of the model. The input str_id is used to select embeddings from the reference set, which are then processed through the model's layers to produce an output.
    """

    def __init__(self, len_coordinates, ref_emb):
        super(SurrogateModel, self).__init__()

        self.emb_dim = ref_emb.shape[1]
        self.len_coordinates = len_coordinates
        self.emb = ref_emb.clone()
        self.emb.requires_grad = False 


        self.conv1 = nn.Conv1d(self.emb_dim, 32, kernel_size=1)
        self.fc1 = nn.Linear(32*self.len_coordinates, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x, emb=True):

        str_emb = self.emb[x]

        x = str_emb.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class AcquisitionFunction(nn.Module):


    def __init__(self, max_dim, len_coordinates, device, tokenizer_surrogate_model):
        super(AcquisitionFunction, self).__init__()
        self.max_dim = max_dim
        self.len_coordinates = len_coordinates
        self.device = device
        self.indices = torch.arange(0, max_dim).long().to(device)
        self.tokenizer_surrogate_model = tokenizer_surrogate_model
        self.str_ids_ignore = []
        self.word_list = self.tokenizer_surrogate_model.batch_decode(
            list(self.tokenizer_surrogate_model.vocab.values())
        )

    def _encode_string(self, string):
        """Encodes a string using the black box tokenizer."""
        return self.tokenizer_surrogate_model.encode(
            string,
            return_tensors="pt",
            max_length=self.len_coordinates,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        ).to(self.device)

    def forward(self, surrogate_model, input_string, coordinate, num_samples):
        

        with torch.no_grad():
        
            str_id = self._encode_string(input_string)
                
            inputs = str_id.repeat(self.max_dim, 1)
            inputs[:, coordinate] = self.indices
            predictions = surrogate_model(inputs).T
 
            top_indices = (
                torch.topk(predictions, num_samples).indices.view(-1).int()
            )

            top_inputs = inputs[top_indices, :]
            top_strings = self.tokenizer_surrogate_model.batch_decode(top_inputs)
            top_strings = top_strings + [input_string]

        return top_strings

