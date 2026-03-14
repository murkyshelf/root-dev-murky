import torch
import torch.nn as nn
import numpy as np
#import matplotlib.pyplot as plt


# physical parameters
m = 1.0
k = 10.0

# neural network
class SpringNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def forward(self,t):
        return self.net(t)


model = SpringNet()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# training points
t = torch.linspace(0,2,100).view(-1,1)
t.requires_grad = True


# synthetic training data
x_true = torch.cos(torch.sqrt(torch.tensor(k/m))*t).detach()

def physics_loss(t):

    x = model(t)

    dx = torch.autograd.grad(
        x,t,
        grad_outputs=torch.ones_like(x),
        create_graph=True
    )[0]

    d2x = torch.autograd.grad(
        dx,t,
        grad_outputs=torch.ones_like(dx),
        create_graph=True
    )[0]

    residual = m*d2x + k*x

    return torch.mean(residual**2)


for epoch in range(8000):

    optimizer.zero_grad()

    x_pred = model(t)

    data_loss = torch.mean((x_pred-x_true)**2)

    phys_loss = physics_loss(t)

    loss = data_loss + 10 * phys_loss

    loss.backward()

    optimizer.step()

    if epoch%500==0:
        print(epoch,loss.item())


# training finished
torch.save(model.state_dict(), "spring_pinn_weights.pth")

print("Model saved")

#Visualization

#t_np = t.detach().numpy()
#_pred = model(t).detach().numpy()
#x_true_np = x_true.detach().numpy()

#plt.plot(t_np,x_true_np,label="True")
#plt.plot(t_np,x_pred,label="PINN")
#plt.legend()
#plt.show()