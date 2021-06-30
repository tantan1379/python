import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


num_time_steps = 50
input_size = 1
hidden_size = 16
output_size = 1
lr = 0.01


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True, # 设置张量的第一个维度为batchsize
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0) # 在张量最前面添加一个新维度
        return out, hidden_prev


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)
hidden_prev = torch.zeros(1, 1, hidden_size)

# Train
for iter in range(3000):
    start = random.randint(0,2)
    time_steps = np.linspace(start, start + 10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)  # 输入序列（0:48）
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)  # 正确输出序列（1:49）

    output, hidden_prev = model(x, hidden_prev)  # x:(1,49,1) batch=1,seq_len=49,feature=1
    hidden_prev = hidden_prev.detach()  # 将得到的hidden_prev作为下一次rnn的输入

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        print(p.grad.norm())
    # torch.nn.utils.clip_grad_norm_(p, 10)
    optimizer.step()

    # if iter % 100 == 0:
    #     print("Iteration: {} loss {}".format(iter, loss.item()))


# Test
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start + 10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)

predictions = []
print(x)
input = x[:, 0, :]
print(input)
for _ in range(x.shape[1]): # x.shape[1]=seq_len(x)
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred
    predictions.append(pred.detach().numpy().ravel()[0])


x = x.data.numpy().ravel()
y = y.data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], predictions)
plt.show()
