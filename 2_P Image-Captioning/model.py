import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embeds = self.embed(captions)
        inp = torch.cat((features.unsqueeze(1), embeds), 1)
        output, self.hidden = self.rnn(inp)
        out = self.linear(output)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        hidden = (torch.randn(self.num_layers,1,self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers,1,self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            output, hidden = self.rnn(inputs,hidden)
            output = output.squeeze(1)  
            output_vocab = self.linear(output)    
            output_word = output_vocab.argmax(1)                
            outputs.append(output_word.item())
            inputs = self.embed(output_word.unsqueeze(0))   
        return outputs