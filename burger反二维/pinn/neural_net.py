#from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pinn.util import log
from pathlib import Path
import pinn.get_points
import time
import matplotlib.pyplot as plt
from math import pi

class MLP(nn.Module):
    
    # Define the MLP创建MLP网络

    def __init__(
        self, pars, device
    ) -> None:

        super().__init__()


        #3维输入，uv输出
        layers = [3,*pars['layers'],2]
        
        # Built the MLP
        modules = []
        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(ResidualBlock(_out))
        
        # Remove last block
        modules.pop()

        self.model = nn.Sequential(*modules)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Forward pass 前向计算
        Y_n = self.model(X)
        #用建立的模型计算预测值
        Y_p = self.particular_solution(X)
        #初始条件的解
        D = self.boundary_distance(X)
        #确保在初始条件上D=0就是特定的解，D是点到边界的距离
        return D * Y_n + (1-D) * Y_p

    def particular_solution(self,X):
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)

        u = torch.sin(2*np.pi*x)*torch.sin(2*np.pi*y)
        v = torch.sin(np.pi*x)*torch.sin(np.pi*y)
        #初始条件的解
        return torch.hstack((u,v))

    def boundary_distance(self,X):
        #距离初始条件的
        alpha = 26.4 # Reaches 0.99 at t = 0.1
        t = X[:,0].reshape(-1, 1)
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)
        dt = torch.tanh(t*alpha)
        dx = 4*x*(1-x)
        dy = 4*y*(1-y)
        return torch.hstack((dt*dx*dy,dt*dx*dy))






class ResidualBlock(nn.Module):

    # Define a block with two layers and a residual connection
    def __init__(self,_size:int):
        super().__init__()



        self.Layer1 = nn.Tanh()
        self.Linear = nn.Linear(_size, _size)
        self.Layer2 = nn.Tanh()
        self.Layer3 = nn.Tanh()
        self.Layer4 = nn.Tanh()
        self.Layer5 = nn.Tanh()


    def forward(self,x):

        ans = x + self.Layer4(self.Linear(self.Layer3(self.Linear(self.Layer2(self.Linear(self.Layer1(x)))))))
        ans1 = x + self.Layer5(self.Linear(self.Layer4(self.Linear(self.Layer3(self.Linear(self.Layer2(self.Linear(self.Layer1(x)))))))))

        return ans







class PINN:
    def __init__(self, nf, ns, pars: dict, device: torch.device = 'cpu') -> None:

        # Parameters
        self.pars = pars
        self.device = device
        self.nf = nf
        self.ns = ns   #精确解的数量
        self.nu = pars['nu']

        self.ls_f1 = torch.tensor(0).to(self.device) #方程u的损失函数
        self.ls_f2 = torch.tensor(0).to(self.device) #方程v的损失函数
        self.ls_f = torch.tensor(0).to(self.device)  #PDE损失函数
        self.ls_s = torch.tensor(0).to(self.device)   #

        # Sample points训练点
        self.sample_points()
        self.zeros = torch.zeros(self.X_f.shape).to(self.device)

        [X_s,Y_s] = pinn.get_points.solution(ns,pars)   #精确解的数量X_s为txy，Y_s为u,v
        self.X_s = torch.tensor(X_s,dtype=torch.float,requires_grad=True).to(self.device)
        self.Y_s = torch.tensor(Y_s,dtype=torch.float,requires_grad=False).to(self.device)

        # Initialize Network
        self.net = MLP(pars,device)
        self.net = self.net.to(device)


        self.loss = nn.MSELoss().to(device)

        #
        self.min_ls_tol = 0.01
        self.min_ls_wait = 10000
        self.min_ls_window = 1000

        self.start_time = time.time()

        self.ls = 0
        self.iter = 0

        self.ls_hist = np.zeros((pars['epochs'],4))

        # Optimizer parameters
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=pars['opt_lr'])


        self.Lambda = []
        # Problem parameter initialization
        # 参数初始化：nu==0或nu==0.03/pi 两种形式都可以试试
        # nu = np.array([0])
        self.nu = np.array([0])
        self.nu = torch.from_numpy(self.nu).float().to(device).requires_grad_(True)
        # 转化nu为张量形式，可以计算梯度
        self.nu.grad = torch.ones((1)).to(device)
        # 梯度为1
        self.optimizer.add_param_group({'params': self.nu, 'lr': 0.00001 })
        # 增加一个nu变量，此参数可以影响方程，以至于影响最终loss，我们要得到loss最小时对应的nu，此时为最佳参数0.00          1
        # 因为我们要求nu精度要求比较高，所以用小的学习率





    def sample_points(self):
        #随机点取值做样本
        X_f = pinn.get_points.domain(self.nf,self.pars)
        self.X_f = torch.tensor(X_f,dtype=torch.float,requires_grad=True).to(self.device)

    def eq_loss(self, X: torch.Tensor):

        # Forward pass
        t = X[:,0].reshape(-1, 1)
        x = X[:,1].reshape(-1, 1)
        y = X[:,2].reshape(-1, 1)
        Y = self.net(torch.hstack((t,x,y)))
        #Y是网络预测出的值
        u = Y[:,0].reshape(-1, 1)
        v = Y[:,1].reshape(-1, 1)

        # Get derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]

        # Compute residuals
        R1 = u_t + u*u_x + v*u_y - self.nu*(u_xx + u_yy)
        R2 = v_t + u*v_x + v*v_y - self.nu*(v_xx + v_yy)


        self.ls_f1 = 0.0001* self.loss(R1,torch.zeros_like(R1))  #方程1损失函数
        self.ls_f2 = 0.0001* self.loss(R2,torch.zeros_like(R1))

        return self.ls_f1 + self.ls_f2

    def sample_loss(self, X: torch.Tensor, Y_gt: torch.Tensor):

        if self.ns == 0:
            return 0

        Y_pred = self.net(X)
        #预测解与真实解的MSE误差
        return self.loss(Y_pred,Y_gt)

    def closure(self) -> torch.nn:

        if self.ns > 0:
            self.ls_s = self.sample_loss(self.X_s, self.Y_s) ##预测解与真实解的MSE误差


        if self.nf > 0:
            self.ls_f = self.eq_loss(self.X_f)  #PDE方程损失函数

        if self.ns > 0 and self.nf > 0:
            self.ls = 1e4 *  self.ls_s + self.ls_f     #PDE方程损失函数与预测精确的和



        elif self.ns > 0:
            self.ls = self.ls_s             #只有预测解
        elif self.nf > 0:
            self.ls = self.ls_f          #只有PDE

        self.optimizer.zero_grad()
        self.ls.backward()

        return self.ls


    def train(self):
        self.net.train()
        loss_history = []
        for i in range(1,self.pars['epochs']):
            self.iter += 1

            self.nu_w = self.nu.item()
            self.Lambda.append([self.nu.item()])


            if self.pars['shuffle'] and i%self.pars['shuffle']==0:
                self.sample_points()
            try:
                self.optimizer.step(self.closure)
            except KeyboardInterrupt:
                print("Stopped by user")
                self.save(0)
                try:
                    input('Press Enter to resume or Ctrl+C again to stop')
                except KeyboardInterrupt:
                    break


            log.info(f'Epoch: {self.iter}, Loss: {self.ls:.3e}, Loss_F: {self.ls_f:.3e} ({self.ls_f1:.3e} + {self.ls_f2:.3e}), Loss_S: {self.ls_s:.3e}, nu: {self.nu_w:.5e}')

            # ls 是PDE损失函数与预测精确MSE的和  ls_f是PDE损失函数   ls_f1 是u函数的  ls_f2是v函数的   ls_s是预测解与真实解的MSE误差
            self.ls_hist[i,:] = torch.hstack((1e-4*self.ls,self.ls_f1,self.ls_f2,self.ls_s)).cpu().detach().numpy()

        #    loss_history.append([self.ls, self.ls_f1, self.ls_f2, self.ls_s])
        #画损失函数图：
        plt.cla()
        plt.plot(self.ls_hist[:,3])
        plt.yscale('log')
        plt.ylim(1e-6, 1)
        plt.legend(('eq loss', 'PDE_u loss', 'PDE_v loss', 'Pred loss'), loc='best')
        plt.savefig('./result_plot/Burger2d_loss.png', bbox_inches='tight', format='png')
        plt.show()



        plt.plot(self.Lambda, label='Pred')
        plt.axhline(y=0.01 / pi, color='b', linestyle='--', label='Reference')
        plt.yscale('log')
        plt.ylim(2e-3, 4e-2)
        plt.legend(loc='best')
        plt.savefig('./result_plot/nu_loss({}).png'.format(40000))
        plt.show()




    def save(self,iter):
        Path(str(self.pars['save_path'].parents[0])).mkdir(parents=True, exist_ok=True) #找父文件
        if iter == -1:
            save_path = self.pars['save_path']
        elif iter == 0:
            save_path = "{0}_partial.{1}".format(*str(self.pars['save_path']).rsplit('.', 1))
        else:
            save_path = f"{str(self.pars['save_path'])[0:-3]}_iter{iter}.pt"

        log.info(f'Saving model to {save_path}')

        ls_hist_temp = self.ls_hist[0:np.nonzero(self.ls_hist[:,0])[0][-1],:]

        torch.save({'model': self.net.state_dict(),'pars':self.pars,'loss':ls_hist_temp, 'time':time.time()-self.start_time, 'memory':torch.cuda.max_memory_allocated(self.device)/(1024*1024)}, save_path)