import torch
import numpy as np
import argparse
import time
from pinn.neural_net import PINN
from pinn.util import log
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pinn.get_points

def get_args():
    parser = argparse.ArgumentParser()
    # 创建解析器
    parser.add_argument('--path',
                        type=str,
                        default='',)
    ##文件路径
    parser.add_argument('--solution',
                        type=str,
                        default='reference_solution.mat',)
    # 缺省为我们的真实解 包含通过模拟解决方案的文件
    parser.add_argument('--comment',
                        type=str,
                        default='',)
    # 要添加到自动生成文件名末尾的字符串
    parser.add_argument('--folder',
                        type=str,
                        default='models',)
    # 保存自动命名模型的文件夹，文件名为models
    parser.add_argument('--resume',
                        type=str,
                        default='',)
    # 模型将被用作初始化
    parser.add_argument('--nf',
                        type=int,
                        default=100000,)
    # f方程（PDE方程）训练点的数量
    parser.add_argument('--ns',
                        type=int,
                        default=10000,)
    # 解的数量
    parser.add_argument('--epochs',
                        type=int,
                        default=70000,)
    # epochs = 100000
    parser.add_argument('--nlayers',
                        type=int,
                        default=4,)
    # 神经网络层数
    parser.add_argument('--nneurons',
                        type=int,
                        default=20,)
    # 每层神经元20个
    parser.add_argument('--shuffle',
                        type=int,
                        default=0,)
    # 每过n轮重新排序样本
    parser.add_argument('--seed',
                        type=int,
                        default=1,)
    # 随机种子
    parser.add_argument('--opt_method',
                        type=str,
                        default='adam',)
    # adam优化算法
    parser.add_argument('--opt_lr',
                        type=float,
                        default=0.001,)
    # 学习率0.01
    parser.add_argument('--loss',
                        type=str,
                        default='mse',)
    # 损失函数类型：mse
    parser.add_argument('--dev',
                        type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),)
    # 运行模型gpu
    args = parser.parse_args()
    
    return args

def main():

    # 定义参数
    pars = dict()
    pars['xi'] = 0 #x最小值
    pars['xf'] = 1
    pars['yi'] = 0
    pars['yf'] = 1
    pars['ti'] = 0
    pars['tf'] = 1
    pars['nu'] = 0.01/np.pi

    # Retrive arguments
    args = get_args()
    nf = args.nf
    ns = args.ns
    pars['epochs'] = args.epochs
    pars['shuffle'] = args.shuffle
    device = args.dev
    resume = args.resume  #初始化
    pars['solution_file'] = args.solution
    

    pars['opt_method'] = args.opt_method
    pars['opt_lr'] = args.opt_lr
    pars['loss_type'] = args.loss



    pars['layers'] = [args.nneurons for i in range(0,args.nlayers)]




    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.path) == 0:
        pars['save_path'] = Path(f'{args.folder}/model_nf{nf}_ns{ns}_MLPRes_2x{args.nlayers}x{args.nneurons}_shuffle{args.shuffle}_seed{args.seed}_{args.opt_method}_lr{args.opt_lr}_loss_{args.loss}{args.comment}.pt')
    else:
        pars['save_path'] = Path(args.path)

    #查看是否存过 如果存过就停
    #if pars['save_path'].is_file():
     #   return


    #模型信息
    log.info(f'Model will be saved to: {pars["save_path"]}')
    log.info(f'Number of samples - Function evaluation: {nf}, Solution samples: {ns}')
    log.info(f'Using device: {device}')

    # Train model
    model = PINN(nf, ns, pars, device)
    #建立模型

    #如果初始化文件不为0就用该文件，
    if len(resume) != 0:
        resume_file = torch.load(resume)
        model.net.load_state_dict(resume_file['model'])


    # 如果模型文件已经存在就load该模型
    if pars['save_path'].is_file():
        #torch.load('model_nf100000_ns10000_MLPRes_2x4x20_shuffle0_seed1_adam_lr0.001_loss_mse.pt')
        model.net.load_state_dict(torch.load(pars['save_path'], map_location="cuda:0"), False)



    #模型训练
    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time



    #画图

    data = loadmat(pars['solution_file'])

    [x, t, y] = np.meshgrid(data['x'], data['t'], data['y'])

    t = t.flatten().reshape(-1, 1)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)

    u = data['uref'].transpose((2, 1, 0)).flatten().reshape(-1, 1)
    v = data['vref'].transpose((2, 1, 0)).flatten().reshape(-1, 1)

    Exactu = np.real(data['uref'])  # real函数返回实部，提取出精确解
    Exactv = np.real(data['vref'])  # real函数返回实部，提取出精确解

    mse_testu = Exactu[:, :, 50].astype(np.float32)  # u精确解提取
    mse_testv = Exactv[:, :, 50].astype(np.float32)  # u精确解提取
    mu_shape = mse_testu.shape
    mv_shape = mse_testv.shape
    t_plot = data['t'].flatten()[:, None]  # 展开成1维度，flatten展0维，[:,none]加一维
    x_plot = data['x'].flatten()[:, None]
    y_plot = data['y'].flatten()[:, None]

    plt.cla()  # 清除axes 画图
    plt.pcolormesh(np.squeeze(x_plot, axis=1), np.squeeze(y_plot, axis=1),
                   mse_testu, cmap='rainbow')
    # squeeze降成一维  .detach阻断反向传播
    # rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
    # 设置颜色组成及对应关系
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("t=0.5 exact_u")
    plt.savefig('./result_plot/Burger2d_exact_u.png', bbox_inches='tight', format='png')
    plt.show()

    plt.cla()  # 清除axes 画图
    plt.pcolormesh(np.squeeze(x_plot, axis=1), np.squeeze(y_plot, axis=1),
                   mse_testv, cmap='rainbow')
    # squeeze降成一维  .detach阻断反向传播
    # rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(0, 1)
    # 设置颜色组成及对应关系
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("t=0.5 exact_v")
    plt.savefig('./result_plot/Burger2d_exact_v.png', bbox_inches='tight', format='png')
    plt.show()

    X1, Y1 = np.meshgrid(x_plot, y_plot)
    X1star = np.hstack((X1.flatten()[:, None], Y1.flatten()[:, None]))
    X1star = X1star.astype(np.float32)  # 数据转换为np
    X1star = torch.from_numpy(X1star).cuda().requires_grad_(True)  # 换成tensor张量 Gpu计算，可求导
    X1 = X1star[:, 1:2]
    Y1 = X1star[:, 0:1]
    x_plot1 = X1.cuda().requires_grad_(True).to(torch.float32)
    y_plot1 = Y1.cuda().requires_grad_(True).to(torch.float32)
    t_plot1 = (0.5 * torch.ones_like(x_plot1)
               ).requires_grad_(True).to(torch.float32)
    uv_plot_1 = model.net(torch.cat([t_plot1, x_plot1, y_plot1], dim=1))
    # .net(torch.hstack((t,x,y))
    u_plot = uv_plot_1[:, 0:1]
    v_plot = uv_plot_1[:, 1:2]

    plt.cla()  # 清除axes 画图
    plt.pcolormesh(np.squeeze(x_plot, axis=1), np.squeeze(y_plot, axis=1),
                   u_plot.cpu().detach().numpy().reshape(mv_shape).T, cmap='rainbow')
    # squeeze降成一维  .detach阻断反向传播
    # rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
    # 设置颜色组成及对应关系
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("t=0.5 predict_u")
    plt.savefig('./result_plot/Burger2d_pretict_u.png', bbox_inches='tight', format='png')
    plt.show()

    plt.cla()  # 清除axes 画图
    plt.pcolormesh(np.squeeze(x_plot, axis=1), np.squeeze(y_plot, axis=1),
                   v_plot.cpu().detach().numpy().reshape(mv_shape).T, cmap='rainbow')
    # squeeze降成一维  .detach阻断反向传播
    # rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(0, 1)
    # 设置颜色组成及对应关系
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("t=0.5 predict_v")
    plt.savefig('./result_plot/Burger2d_pretict_v.png', bbox_inches='tight', format='png')
    plt.show()

    plt.cla()  # 清除axes 画图
    plt.pcolormesh(np.squeeze(x_plot, axis=1), np.squeeze(y_plot, axis=1),
                   mse_testu - u_plot.cpu().detach().numpy().reshape(mv_shape).T, cmap='rainbow')
    # squeeze降成一维  .detach阻断反向传播
    # rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(0, 1)
    # 设置颜色组成及对应关系
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("t=0.5 point_error_u")
    plt.savefig('./result_plot/Burger2d_point_error_u.png', bbox_inches='tight', format='png')
    plt.show()

    plt.cla()  # 清除axes 画图
    plt.pcolormesh(np.squeeze(x_plot, axis=1), np.squeeze(y_plot, axis=1),
                   mse_testv - v_plot.cpu().detach().numpy().reshape(mv_shape).T, cmap='rainbow')
    # squeeze降成一维  .detach阻断反向传播
    # rainbow颜色显示具体函数值
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(0, 1)
    # 设置颜色组成及对应关系
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("t=0.5 point_error_v")
    plt.savefig('./result_plot/Burger2d_point_error_v.png', bbox_inches='tight', format='png')
    plt.show()

    # t=0.5 x=0.15 预测解与真实解
    # 六幅子图，分别是
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    x_in = 0.15
    t_in = 0.5
    [X_s, Y_s] = pinn.get_points.solution_plotx(t_in, x_in, pars)  # 精确解的数量X_s为txy，Y_s为u,v
    X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(device)

    x_plot = X_s[:, 2].detach().cpu().numpy()
    Y_pred = model.net(X_s)
    u_15_exc = Y_s[:, 0:1]
    v_15_exc = Y_s[:, 1:2]
    u_15_pre = Y_pred[:, 0:1]
    v_15_pre = Y_pred[:, 1:2]

    ax[0].plot(x_plot, u_15_exc, 'b-', linewidth=2)  # 蓝线是精确解
    ax[0].plot(x_plot, u_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[0].plot(x_plot, v_15_exc, 'b-.', linewidth=2)  # 蓝线是精确解
    ax[0].plot(x_plot, v_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-.', lw='2')  # 红是预测
    ax[0].set_xlabel('y')
    ax[0].set_ylabel('u_v(y)')
    ax[0].axis('square')
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([-0.3, 1])
    ax[0].set_title('t = 0.5 x=0.15', fontsize=10)

    # 第二幅子图
    x_in = 0.33
    t_in = 0.5
    [X_s, Y_s] =  pinn.get_points.solution_plotx(t_in, x_in, pars)  # 精确解的数量X_s为txy，Y_s为u,v
    X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(device)
    x_plot = X_s[:, 2].detach().cpu().numpy()
    Y_pred = model.net(X_s)
    u_15_exc = Y_s[:, 0:1]
    v_15_exc = Y_s[:, 1:2]
    u_15_pre = Y_pred[:, 0:1]
    v_15_pre = Y_pred[:, 1:2]

    ax[1].plot(x_plot, u_15_exc, 'b-', linewidth=2)  # 蓝线是精确解
    ax[1].plot(x_plot, u_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[1].plot(x_plot, v_15_exc, 'b-.', linewidth=2)  # 蓝线是精确解
    ax[1].plot(x_plot, v_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-.', lw='2')  # 红是预测
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('u_v(y)')
    ax[1].axis('square')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([-0.3, 1])
    ax[1].set_title('t = 0.5 x=0.33', fontsize=10)

    # 第三幅子图
    x_in = 0.4
    t_in = 0.5
    [X_s, Y_s] =  pinn.get_points.solution_plotx(t_in, x_in, pars)  # 精确解的数量X_s为txy，Y_s为u,v
    X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(device)
    x_plot = X_s[:, 2].detach().cpu().numpy()
    Y_pred = model.net(X_s)
    u_15_exc = Y_s[:, 0:1]
    v_15_exc = Y_s[:, 1:2]
    u_15_pre = Y_pred[:, 0:1]
    v_15_pre = Y_pred[:, 1:2]

    ax[2].plot(x_plot, u_15_exc, 'b-', linewidth=2)  # 蓝线是精确解
    ax[2].plot(x_plot, u_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[2].plot(x_plot, v_15_exc, 'b-.', linewidth=2)  # 蓝线是精确解
    ax[2].plot(x_plot, v_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-.', lw='2')  # 红是预测
    ax[2].set_xlabel('y')
    ax[2].set_ylabel('u_v(y)')
    ax[2].axis('square')
    ax[2].set_xlim([0, 1])
    ax[2].set_ylim([-0.3, 1])
    ax[2].set_title('t = 0.5 x=0.4', fontsize=10)
    
    plt.show()



    #按y画图
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    y_in = 0.53
    t_in = 0.5
    [X_s, Y_s] = pinn.get_points.solution_ploty(t_in, y_in, pars)  # 精确解的数量X_s为txy，Y_s为u,v
    X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(device)

    x_plot = X_s[:, 1].detach().cpu().numpy()
    Y_pred = model.net(X_s)
    u_15_exc = Y_s[:, 0:1]
    v_15_exc = Y_s[:, 1:2]
    u_15_pre = Y_pred[:, 0:1]
    v_15_pre = Y_pred[:, 1:2]

    ax[0].plot(x_plot, u_15_exc, 'b-', linewidth=2)  # 蓝线是精确解
    ax[0].plot(x_plot, u_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[0].plot(x_plot, v_15_exc, 'b-.', linewidth=2)  # 蓝线是精确解
    ax[0].plot(x_plot, v_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-.', lw='2')  # 红是预测
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('u_v(x)')
    ax[0].axis('square')
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([-1, 1])
    ax[0].set_title('t = 0.5 y=0.53', fontsize=10)

    # 第二幅子图
    y_in = 0.76
    t_in = 0.5
    [X_s, Y_s] = pinn.get_points.solution_ploty(t_in, y_in, pars)  # 精确解的数量X_s为txy，Y_s为u,v
    X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(device)
    x_plot = X_s[:, 1].detach().cpu().numpy()
    Y_pred = model.net(X_s)
    u_15_exc = Y_s[:, 0:1]
    v_15_exc = Y_s[:, 1:2]
    u_15_pre = Y_pred[:, 0:1]
    v_15_pre = Y_pred[:, 1:2]

    ax[1].plot(x_plot, u_15_exc, 'b-', linewidth=2)  # 蓝线是精确解
    ax[1].plot(x_plot, u_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[1].plot(x_plot, v_15_exc, 'b-.', linewidth=2)  # 蓝线是精确解
    ax[1].plot(x_plot, v_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-.', lw='2')  # 红是预测
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('u_v(x)')
    ax[1].axis('square')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([-1, 1])
    ax[1].set_title('t = 0.5 y=0.76', fontsize=10)

    # 第三幅子图
    y_in = 0.87
    t_in = 0.5
    [X_s, Y_s] = pinn.get_points.solution_ploty(t_in, y_in, pars)  # 精确解的数量X_s为txy，Y_s为u,v
    X_s = torch.tensor(X_s, dtype=torch.float, requires_grad=True).to(device)
    x_plot = X_s[:, 1].detach().cpu().numpy()
    Y_pred = model.net(X_s)
    u_15_exc = Y_s[:, 0:1]
    v_15_exc = Y_s[:, 1:2]
    u_15_pre = Y_pred[:, 0:1]
    v_15_pre = Y_pred[:, 1:2]
    inx = np.arange(x_in * 100 * 101 + 101 * 101 * t_in * 100, x_in * 100 * 101 + 101 * 101 * t_in * 100 + 101).astype(
        'int64')
    ax[2].plot(x_plot, u_15_exc, 'b-', linewidth=2)  # 蓝线是精确解
    ax[2].plot(x_plot, u_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-', lw='2')  # 红是预测
    ax[2].plot(x_plot, v_15_exc, 'b-.', linewidth=2)  # 蓝线是精确解
    ax[2].plot(x_plot, v_15_pre.reshape((-1, 1)).detach().cpu().numpy(), 'r-.', lw='2')  # 红是预测
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('u_v(x)')
    ax[2].axis('square')
    ax[2].set_xlim([0, 1])
    ax[2].set_ylim([-1, 1])
    ax[2].set_title('t = 0.5 y=0.87', fontsize=10)

    plt.show()


    log.info(f'Training time: {elapsed:.4f}s')
    model.save(-1)
    log.info('Finished training.')





if __name__== "__main__":
    main()