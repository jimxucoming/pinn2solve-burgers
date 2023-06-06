from net import Net
import torch
from torch.autograd import grad
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from math import pi
import time

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]
    #gradoutputs 向量与f中的自变量有关   create_graph可以计算高阶导数


def PDE(u, t, x, nu):
    return d(u, t) + u * d(u, x) - nu * d(d(u, x), x)
    #PDE函数u_t + u*u_x -nu*u_xx 应该趋于0

def train():
    # Parameter setting
    #准备工作
    nu = 0.01 / pi
    lr = 0.001    #学习率
    epochs = 30000   #12000轮周期
    t_left, t_right = 0., 1.   #边界t在0-1
    x_left, x_right = -1., 1.   #x在-1 -1


    n_f, n_b_1, n_b_2 = 12000, 6000, 6000   #训练点数量 残差f:10000个 边界：10000个 共20000



    # test data
    data = scipy.io.loadmat('./result/burgers_shock.mat')
    #有三个double数据，分别是
    #x :256*1 是-1到1的均分
    #t :100*1 0-1
    #usol精确解 256*100
    Exact = np.real(data['usol']).T  #real函数返回实部，提取出精确解 加转置是之后按行展开对应解
    t = data['t'].flatten()[:, None]  #展开成1维度，flatten展0维，[:,none]加一维
    x = data['x'].flatten()[:, None]
    X, T = np.meshgrid(x, t)  #生成二维网格
    s_shape = X.shape   #100行256列
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # 在水平方向上平铺(25600, 2) hstack按行加，X flatten 100*256  T同理
    X_star = X_star.astype(np.float32) #数据转换为np
    X_star = torch.from_numpy(X_star).cuda().requires_grad_(True)  #换成tensor张量 Gpu计算，可求导
    u_star = Exact.flatten()[:, None] #精确解
    u_star = u_star.astype(np.float32)
    u_star = torch.from_numpy(u_star).cuda().requires_grad_(True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    #加载gpu0



    PINN = Net(seq_net=[2, 20, 20, 20, 20, 20, 20,  1], activation=torch.tanh).to(device)
    #pinn网络设置 指定到gpu上运算




    optimizer = torch.optim.Adam(PINN.parameters(), lr)
    #优化器
    criterion = torch.nn.MSELoss()
    #残差函数 均方误差 预测值与真实值用均方误差

    loss_history = []
    #损失函数历史
    mse_loss = []
    #均方误差损失函数

    plt.cla()  #清除axes 画图
    mse_test = u_star #残差精确测试项提取
    '''
    精确解画图
    '''
    plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
                   mse_test.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
    #squeeze降成一维  .detach阻断反向传播
    #rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
    #设置颜色组成及对应关系
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig('./result_plot/Burger1d_exact.png', bbox_inches='tight', format='png')
    #存图片
    #
    plt.show()
    #

    start_time = time.time()



    #接下来是循环学习
    for epoch in range(epochs):
        optimizer.zero_grad()
        # 把loss关于weight归0
        t_f = ((t_left + t_right) / 2 + (t_right - t_left) *
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        #随机点t属于[0,1] 均匀分布 用n_f=10000个点  属于pde内部随机选点

        x_f = ((x_left + x_right) / 2 + (x_right - x_left) *
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        ##随机点x属于[-1,1] 均匀分布 用n_f=10000个点 属于pde内部随机选点

        u_f = PINN(torch.cat([t_f, x_f], dim=1))
        #内部点pinn得到的函数值

        PDE_ = PDE(u_f, t_f, x_f, nu)
        #之前写的pde函数
        mse_PDE = criterion(PDE_, torch.zeros_like(PDE_))
        #求PDE算出来的损失函数，PDE_要尽可能趋近与0，及内部的点要符合PDE方程，均方误差
        # boundary
        #边界条件也要计算损失函数
        x_rand = ((x_left + x_right) / 2 + (x_right - x_left) *
                  (torch.rand(size=(n_b_1, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
            #随机点x属于[-1,1] 均匀分布 用n_b_1=5000个点 属于pde边界点t=0
        t_b = (t_left * torch.ones_like(x_rand)
               ).requires_grad_(True)
            #t_b维度上对齐x_rand，属于边界点t=0
        u_b_1 = PINN(torch.cat([t_b, x_rand], dim=1)) + torch.sin(pi * x_rand)
             #t=0,x=[-1,1]上边界点的函数值，要接近-sin(x*pi),现在要趋近于0
        t_rand = ((t_left + t_right) / 2 + (t_right - t_left) *
                  (torch.rand(size=(n_b_2, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
            #随机点t属于[0,1] 均匀分布 用n_b_2=5000个点  属于pde边界随机选点
        x_b_1 = (x_left * torch.ones_like(t_rand)
                 ).requires_grad_(True)
        x_b_2 = (x_right * torch.ones_like(t_rand)
                 ).requires_grad_(True)
            #两个边界条件x=-1与x=1
        u_b_2 = PINN(torch.cat([t_rand, x_b_1], dim=1))
            #左边界的函数值，应该趋近于0
        u_b_3 = PINN(torch.cat([t_rand, x_b_2], dim=1))
            #右边界的函数值，应该趋近与0
        mse_BC_1 = criterion(u_b_1, torch.zeros_like(u_b_1))
        mse_BC_2 = criterion(u_b_2, torch.zeros_like(u_b_2))
        mse_BC_3 = criterion(u_b_3, torch.zeros_like(u_b_3))
        #边界条件的损失函数
        mse_BC = mse_BC_1 + mse_BC_2 + mse_BC_3

        # loss函数综合方程内部与边界条件，权重比例为1：1
        loss = 1 * mse_PDE + 1 * mse_BC


        # 预测解与真实解的差距
        x_pred = X_star[:, 0:1]
        t_pred = X_star[:, 1:2]
        u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))
        mse_test = criterion(u_pred, u_star)
        #记录损失函数历史
        loss_history.append([mse_PDE.item(), mse_BC.item(), mse_test.item()])
        mse_loss.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},  loss: {:.08e}, mse_loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), loss.item(), mse_test.item()
                )
            )
        #1000个epoch一轮报告数据，依次是PDE方程的损失函数，边界条件的损失函数，总的损失函数
        loss.backward()
        # 反向传播求梯度
        optimizer.step()
        # 更新所有参数
        
        '''
        # 画图2000一轮观察其训练情况
        if (epoch + 1) % 2000 == 0:
            plt.cla()
            fig = plt.figure(figsize=(12, 4))
            #初始画布
            xx = torch.linspace(0, 1, 1000).cpu()
            yy = torch.linspace(-1, 1, 1000).cpu()
            x1, y1 = torch.meshgrid([xx, yy])
            #划分网格
            s1 = x1.shape
            x1 = x1.reshape((-1, 1))
            y1 = y1.reshape((-1, 1))
            #换成一列数据
            out = torch.cat([x1, y1], dim=1).to(device)
            z = PINN(out)
            #将这些数据点导入PINN模型
            z_out = z.reshape(s1)
            #转成1000000行1列（1000*1000）
            out = z_out.cpu().T.detach().numpy()
            #作图准备完成
            #画的是内部的点（均匀分布）
            plt.pcolormesh(xx, yy, out, cmap='rainbow')
            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.mappable.set_clim(-1, 1)  #用来控制颜色缩放的
            plt.xticks([])
            plt.yticks([])
            # plt.savefig('./result_plot/Burger1d_pred_{}.png'.format(epoch + 1), bbox_inches='tight', format='png')
            plt.show()

            #按照给的精确解的点来画
            x_pred = X_star[:, 0:1]
            t_pred = X_star[:, 1:2]
            u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))
            plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
                           u_pred.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.mappable.set_clim(-1, 1)
            plt.xticks([])
            plt.yticks([])
            plt.savefig('./result_plot/Burger1d_pred_{}.png'.format(epoch + 1), bbox_inches='tight', format='png')
            plt.show()

            #预测解与真实解差距
            plt.cla()
            mse_test = abs(u_pred - u_star)#求差的绝对值
            plt.pcolormesh(np.squeeze(t, axis=1), np.squeeze(x, axis=1),
                           mse_test.cpu().detach().numpy().reshape(s_shape).T, cmap='rainbow')
            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.mappable.set_clim(0, 0.3)
            plt.xticks([])
            plt.yticks([])
            plt.savefig('./result_plot/Burger1d_error_{}.png'.format(epoch + 1), bbox_inches='tight', format='png')
            plt.show()


            #三幅子图，分别是t=0.25,t=0.5,t=0.75时的x图
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            x_25 = x.astype(np.float32)
            x_25 = torch.from_numpy(x_25).cuda().requires_grad_(True)
            t_25 = (0.25 * torch.ones_like(x_25)).requires_grad_(True)
            u_25 = PINN(torch.cat([t_25, x_25], dim=1))
            ax[0].plot(x, Exact[25, :], 'b-', linewidth=2)#蓝线是精确解
            ax[0].plot(x, u_25.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')#红是预测
            ax[0].legend(labels=['Exact','PINN'])
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('u(t,x)')
            ax[0].axis('square')
            ax[0].set_xlim([-1.1, 1.1])
            ax[0].set_ylim([-1.1, 1.1])
            ax[0].set_title('t = 0.25', fontsize=10)

            #第二幅子图
            t_50 = (0.5 * torch.ones_like(x_25)).requires_grad_(True)
            u_50 = PINN(torch.cat([t_50, x_25], dim=1))
            ax[1].plot(x, Exact[50, :], 'b-', linewidth=2)
            ax[1].plot(x, u_50.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2)
            ax[1].legend(labels=['Exact', 'PINN'])
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('u(t,x)')
            ax[1].axis('square')
            ax[1].set_xlim([-1.1, 1.1])
            ax[1].set_ylim([-1.1, 1.1])
            ax[1].set_title('t = 0.50', fontsize=10)
            #ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)

            # 第三幅子图
            t_75 = (0.75 * torch.ones_like(x_25)).requires_grad_(True)
            u_75 = PINN(torch.cat([t_75, x_25], dim=1))
            ax[2].plot(x, Exact[75, :], 'b-', linewidth=2)
            ax[2].plot(x, u_75.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2)
            ax[2].legend(labels=['Exact', 'PINN'])
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('u(t,x)')
            ax[2].axis('square')
            ax[2].set_xlim([-1.1, 1.1])
            ax[2].set_ylim([-1.1, 1.1])
            ax[2].set_title('t = 0.75', fontsize=10)
            plt.savefig('./result_plot/Burger1d_t_{}.png'.format(epoch + 1), bbox_inches='tight', format='png')
            plt.show()
        '''

    elapsed = time.time() - start_time
    print(elapsed)

    #最终结果作图
    plt.cla()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.ylim(1e-5, 1)
    plt.legend(('PDE loss', 'BC loss', 'Pred loss'), loc='best')
    plt.savefig('./result_plot/Burger1d_loss_{}.png'.format(epochs + 1), bbox_inches='tight', format='png')


    # 三幅子图，分别是t=0.25,t=0.5,t=0.75时的x图
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    x_25 = x.astype(np.float32)
    x_25 = torch.from_numpy(x_25).cuda().requires_grad_(True)
    t_25 = (0.25 * torch.ones_like(x_25)).requires_grad_(True)
    u_25 = PINN(torch.cat([t_25, x_25], dim=1))
    ax[0].plot(x, Exact[25, :], 'b-', linewidth=2)  # 蓝线是精确解
    ax[0].plot(x, u_25.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[0].legend(labels=['Exact', 'PINN'])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('u(t,x)')
    ax[0].axis('square')
    ax[0].set_xlim([-1.1, 1.1])
    ax[0].set_ylim([-1.1, 1.1])
    ax[0].set_title('t = 0.25', fontsize=10)
    # 第二幅子图
    t_50 = (0.5 * torch.ones_like(x_25)).requires_grad_(True)
    u_50 = PINN(torch.cat([t_50, x_25], dim=1))
    ax[1].plot(x, Exact[50, :], 'b-', linewidth=2)
    ax[1].plot(x, u_50.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2)
    ax[1].legend(labels=['Exact', 'PINN'])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('u(t,x)')
    ax[1].axis('square')
    ax[1].set_xlim([-1.1, 1.1])
    ax[1].set_ylim([-1.1, 1.1])
    ax[1].set_title('t = 0.50', fontsize=10)
    # ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
    # 第三幅子图
    t_75 = (0.75 * torch.ones_like(x_25)).requires_grad_(True)
    u_75 = PINN(torch.cat([t_75, x_25], dim=1))
    ax[2].plot(x, Exact[75, :], 'b-', linewidth=2)
    ax[2].plot(x, u_75.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2)
    ax[2].legend(labels=['Exact', 'PINN'])
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('u(t,x)')
    ax[2].axis('square')
    ax[2].set_xlim([-1.1, 1.1])
    ax[2].set_ylim([-1.1, 1.1])
    ax[2].set_title('t = 0.75', fontsize=10)


    plt.show()


if __name__ == '__main__':
    train()
