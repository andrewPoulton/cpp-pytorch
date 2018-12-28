import torch

class MyModule(torch.jit.ScriptModule):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.emb = torch.nn.Embedding(20, M)
        self.weight = torch.nn.Linear(M, N)

    @torch.jit.script_method
    def forward(self, input):
      # print(input.size())
      input = self.emb(input).squeeze()
      # print(input.size())
      if bool(input.sum() > 0):
        output = self.weight(input)
      else:
        output = self.weight.weight + input
      return output


if __name__ == "__main__":
  N, M = [10,5]
  u = torch.randint(20, (N,))
  m = MyModule(N,M)
  print(m(u))
  m.save('model.pt')