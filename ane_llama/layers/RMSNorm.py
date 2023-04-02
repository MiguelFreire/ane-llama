import torch
import torch.nn as nn    
#https://arxiv.org/abs/1910.07467
class RMSNormANE(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.expected_rank = len('BC1S')
        self.hidden_size = hidden_size

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps
        
    def forward(self, hidden_states):
        input_rank = len(hidden_states.size())

        if input_rank == 3 and hidden_states.size(2) == self.hidden_size:
            hidden_states = hidden_states.transpose(1,2).unsqueeze(2)

        assert input_rank == self.expected_rank
        assert hidden_states.size(1) == self.hidden_size
        #do we really need to convert to f32 here? 
        var  = hidden_states.to(torch.float32).pow(2).mean(1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(var + self.variance_eps)

        return self.weight.view(1, self.hidden_size, 1, 1) * hidden_states
    
    