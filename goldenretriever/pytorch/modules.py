import torch


class MatrixMultiplicationModule(torch.nn.Module):
    def __init__(self, embeddings, requires_grad: bool = False):
        super().__init__()
        self.embeddings = torch.nn.Parameter(embeddings, requires_grad=requires_grad)

    def forward(self, query):
        return torch.matmul(query, self.embeddings.T)
