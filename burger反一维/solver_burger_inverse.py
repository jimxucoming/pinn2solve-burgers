from net import Net
import os
import matplotlib.pyplot as plt
from Parser_PINN import get_parser
import scipy.io
import numpy as np
import torch
from torch.autograd import grad
from math import pi



def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]
#定义微分函数，gradoutputs 向量与f中的自变量有关   create_graph可以计算高阶导数


def PDE(u, t, x, nu):
    return d(u, t) + u * d(u, x) - nu * d(d(u, x), x)
#定义PDE方程u_t + u*u_x -nu*u_xx 应该趋于0，
# 目的是求nu的值，nu的真实值为0.01/pi

########

def train(args):
    # nu = 0.01/pi
    #训练
    #args:parser = get_parser()  args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    # 加载gpu0

    PINN = Net(seq_net=args.seq_net, activation=args.activation).to(device)
    #建立网络：seq_net=[2, 20, 20, 20, 20, 20, 20, 1]

    optimizer = args.optimizer(PINN.parameters(), args.lr)
    #设置优化器，参数为net的参数：w权重和b偏置，之后会加入nu变量，学习率为lr=0.001，



    # Problem parameter initialization
    #参数初始化：nu==0或nu==0.03/pi 两种形式都可以试试
    nu = np.array([0])
    # nu = np.array([0.03/pi])
    nu = torch.from_numpy(nu).float().to(device).requires_grad_(True)
    #转化nu为张量形式，可以计算梯度
    nu.grad = torch.ones((1)).to(device)
    #梯度为1

    optimizer.add_param_group({'params': nu, 'lr': 0.00001})
    #增加一个nu变量，此参数可以影响方程，以至于影响最终loss，我们要得到loss最小时对应的nu，此时为最佳参数
    #因为我们要求nu精度要求比较高，所以用小的学习率




    # test data精确解或者可以用精度较高的实验数据，这里我们用scipy的数据
    data = scipy.io.loadmat('Data/burgers_shock.mat')
    #有三个double数据，分别是
    #x :256*1 是-1到1的均分
    #t :100*1 0-1
    #usol精确解 256*100
    Exact = np.real(data['usol']).T
    #精确解提取，#real函数返回实部，提取出精确解 加转置是之后按行展开对应解
    t = data['t'].flatten()[:, None] #展开成1维度，flatten展0维，[:,none]加一维
    x = data['x'].flatten()[:, None]
    X, T = np.meshgrid(x, t)  #生成二维网格
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # 在水平方向上平铺(25600, 2)
    # hstack按行加，X flatten 100*256  T同理
    X_star = X_star.astype(np.float32)  # 第一列为x,第二列为t   #数据转换为float
    X_star = torch.from_numpy(X_star).cuda().requires_grad_(True)   #换成tensor张量 Gpu计算，可求导
    u_star = Exact.flatten()[:, None]   #精确解
    u_star = u_star.astype(np.float32)
    u_star = torch.from_numpy(u_star).cuda().requires_grad_(True)

    # data：从上面25600个精确解中抽取10000个不同的随机数据点
    N_train = 10000  #训练点的数量为10000
    N, T = 256, 100
    idx = np.random.choice(N * T, N_train, replace=False)
    #从n*t中提取n_train个数据，replace=false指不可相同
    x_train = X_star[idx, 0:1].requires_grad_(True)
    t_train = X_star[idx, 1:2].requires_grad_(True)
    u_train = u_star[idx]

    loss_history = []
    # 损失函数历史（PDE bc 等）
    Lambda = []
    #nu的历史数值
    Loss_history = []
    #loss函数的历史值
    test_loss = []
    #10000个测试点历史值
    # 接下来是循环学习
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        #把loss关于weight归0
        t_f = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        # 随机点t属于[0,1] 均匀分布 用args.n_f=10000个点  属于pde内部随机选点
        x_f = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
               (torch.rand(size=(args.n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        ##随机点x属于[-1,1] 均匀分布 用args.n_f=10000个点 属于pde内部随机选点
        u_f = PINN(torch.cat([t_f, x_f], dim=1))
        # 内部点pinn得到的函数值
        PDE_ = PDE(u_f, t_f, x_f, nu)
        # 之前写的pde函数
        mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))
        # 求PDE算出来的损失函数，PDE_要尽可能趋近与0，及内部的点要符合PDE方程，均方误差
        # Boundary
        #边界条件也要计算损失函数
        x_rand = ((args.x_left + args.x_right) / 2 + (args.x_right - args.x_left) *
                  (torch.rand(size=(args.n_b_1, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
             # 随机点x属于[-1,1] 均匀分布 用n_b_1=400个点 属于pde边界点t=0
        t_b = (args.t_left * torch.ones_like(x_rand)
               ).requires_grad_(True)
             # t_b维度上对齐x_rand，属于边界点t=0
        u_b_1 = PINN(torch.cat([t_b, x_rand], dim=1)) + torch.sin(pi * x_rand)
            # t=0,x=[-1,1]上边界点的函数值，要接近-sin(x*pi),现在要趋近于0
        t_rand = ((args.t_left + args.t_right) / 2 + (args.t_right - args.t_left) *
                  (torch.rand(size=(args.n_b_2, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
            # 随机点t属于[0,1] 均匀分布 用n_b_2=400个点  属于pde边界随机选点
        x_b_1 = (args.x_left * torch.ones_like(t_rand)
                 ).requires_grad_(True)
        x_b_2 = (args.x_right * torch.ones_like(t_rand)
                 ).requires_grad_(True)
            # 两个边界条件x=-1与x=1
        u_b_2 = PINN(torch.cat([t_rand, x_b_1], dim=1))
        # 左边界的函数值，应该趋近于0
        u_b_3 = PINN(torch.cat([t_rand, x_b_2], dim=1))
        # 右边界的函数值，应该趋近与0

        mse_BC_1 = args.criterion(u_b_1, torch.zeros_like(u_b_1))
        mse_BC_2 = args.criterion(u_b_2, torch.zeros_like(u_b_2))
        mse_BC_3 = args.criterion(u_b_3, torch.zeros_like(u_b_3))
        # 边界条件的损失函数
        mse_BC = mse_BC_1 + mse_BC_2 + mse_BC_3

        # Data，精确解损失函数，10000个数据
        u_data = PINN(torch.cat([t_train, x_train], dim=1))
        mse_Data = args.criterion(u_data, u_train)
        #精确解u_train与预测解u_data之间的均方误差

        # loss函数综合方程内部与边界条件与精确解条件，权重比例为1：1：1
        loss = args.PDE_panelty * mse_PDE + args.BC_panelty * mse_BC + args.Data_panelty*mse_Data

        # Pred loss 预测解与真实解的差距 25600个数据
        x_pred = X_star[:, 0:1]
        t_pred = X_star[:, 1:2]
        u_pred = PINN(torch.cat([t_pred, x_pred], dim=1))
        mse_test = args.criterion(u_pred, u_star)
        #记录损失函数历史
        loss_history.append([mse_PDE.item(), mse_BC.item(), mse_Data.item(), mse_test.item()])
        #包括pde,bc,data,test所有的损失函数
        Lambda.append([nu.item()])
        #nu 历史值
        Loss_history.append([loss.item()])
        #loss 历史值
        test_loss.append([mse_test.item()])
        #test历史值

        loss.backward(retain_graph=True)
        # 反向传播求梯度
        optimizer.step()
        # 更新所有参数

        if (epoch + 1) % 1000 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},  loss: {:.08e}'.format(
                    epoch+1, mse_PDE.item(), mse_BC.item(), loss.item()
                )
            )
            print(nu.item())
        # 1000个epoch一轮报告数据，依次是PDE方程的损失函数，边界条件的损失函数，总的损失函数，还有nu的值


        #画图2000一轮观察其训练情况
        if (epoch + 1) % 20000 == 0:
            # plt.cla()
            fig = plt.figure(figsize=(12, 4))
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
            # 将这些数据点导入PINN模型
            z_out = z.reshape(s1)
            # 转成1000000行1列（1000*1000）
            out = z_out.cpu().T.detach().numpy()
            # out = z_out.cpu().T.detach().numpy()[::-1,:]
            # 作图准备完成
            # 画的是内部的点（均匀分布）
            plt.pcolormesh(xx, yy, out, cmap='rainbow')
            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.mappable.set_clim(-1, 1)  #用来控制颜色缩放的
            plt.xlabel('t')
            plt.ylabel('x')
            plt.xlim(0, 1)  # x坐标轴刻度值范围
            plt.ylim(-1, 1)  # y坐标轴刻度值范围
            plt.xticks([])
            plt.yticks([])

            plt.savefig('./result_plot/PINN_pred({}).png'.format(args.epochs))
            plt.show()

            #三幅子图，分别是t=0.25,t=0.5,t=0.75时的x图
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            x_25 = x.astype(np.float32)
            x_25 = torch.from_numpy(x_25).cuda().requires_grad_(True)
            t_25 = (0.25 * torch.ones_like(x_25)).requires_grad_(True)
            u_25 = PINN(torch.cat([t_25, x_25], dim=1))
            ax[0].plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
            ax[0].plot(x, u_25.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2, label='PINN')
            ax[0].legend(labels=['Exact','PINN'])
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('u(t,x)')
            ax[0].axis('square')
            ax[0].set_xlim([-1.1, 1.1])
            ax[0].set_ylim([-1.1, 1.1])
            ax[0].set_title('t = 0.25', fontsize=10)
            # ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

            t_50 = (0.5 * torch.ones_like(x_25)).requires_grad_(True)
            u_50 = PINN(torch.cat([t_50, x_25], dim=1))
            ax[1].plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
            ax[1].plot(x, u_50.reshape((-1, 1)).detach().cpu().numpy(), 'r--', linewidth=2, label='PINN')
            ax[1].legend(labels=['Exact', 'PINN'])
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('u(t,x)')
            ax[1].axis('square')
            ax[1].set_xlim([-1.1, 1.1])
            ax[1].set_ylim([-1.1, 1.1])
            ax[1].set_title('t = 0.50', fontsize=10)
            ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)

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

            plt.savefig('./result_plot/PINN({}).png'.format(args.epochs))
            plt.show()

        
    #最终结果作图
    #损失函数
    plt.cla()
    plt.plot(loss_history)
    plt.yscale('log')
    # plt.ylim(1e-4, 1e-1)
    plt.legend(('PDE loss', 'BC loss', 'Data loss', 'Pred loss'), loc='best')
    plt.savefig('./result_plot/loss({}).png'.format(args.epochs))
    plt.show()

    #nu的值
    plt.plot(Lambda, label='Pred')
    plt.axhline(y=0.01 / pi, color='b', linestyle='--', label='Reference')
    plt.yscale('log')
    plt.ylim(1e-4, 1e-2)
    plt.legend(loc='best')
    plt.savefig('./result_plot/nu_loss({}).png'.format( args.epochs))
    plt.show()


    #np.save('./result_data/nu({}_2).npy'.format(args.epochs), Lambda)
    #np.save('./result_data/nu({}).npy'.format(args.epochs), Lambda)
    #np.save('./result_data/test_loss({}).npy'.format(args.epochs), test_loss)
    #np.save('./result_data/training_loss({}).npy'.format(args.epochs), Loss_history)
    torch.save(PINN.state_dict(), './result_data/PINN({}).pth'.format(args.epochs))
    #保存模型文件



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args)
