from anomalia.resmed.preprocess import *


dataset = read_data("data/resmed/staging/BBett_idle/", "config/resmed.json")



test = torch.rand((3,2,4))
test2 = torch.rand((4,2,4))

torch.cat([test, test2])