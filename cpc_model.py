import torch

class CPCModel(torch.nn.Module):
    def __init__(self, encoder, AR):
        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batch_data):
        encoder_output = self.gEncoder(batch_data)
        context_input = encoder_output.permute(0, 2, 1)
        context_output = self.gAR(context_input)

        return context_output, encoder_output

class CharacterClassifier(torch.nn.Module):
  def __init__(self,
               input_dim : int,
               n_characters : int):
    super(CharacterClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, n_characters)
    
  def forward(self, x):
    return self.linear(x)