import torch
import torch.distributions as dist

loc = torch.zeros((8, 3))
scale = torch.ones((8, 3)) / 0.5

dst = dist.Laplace(loc, scale)


value = torch.randn((8, 3)) * 100

- dst.log_prob(value)


-(torch.log(2 * scale) - torch.abs(value - loc) / scale)



def test(a):
    print(a[0], a[1])


