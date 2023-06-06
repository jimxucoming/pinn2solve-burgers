import argparse
import torch
#argarse可以让人更好的编写命令行接口，直接修改参数
#通过ArgumentParser()创建解析器对象
#通过add_argument()方法添加程序参数信息
#通过parse_args()方法解析参数

def get_parser():
    parser = argparse.ArgumentParser()
    #添加参数例如：
    # parser.add_argument(
    #     '--seq_net', default=[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # )
    parser.add_argument(
        '--seq_net', default=[2, 20, 20, 20, 20, 20, 20, 1]
        #parser.seq_net 默认值为[2, 20, 20, 20, 20, 20, 20, 1]
        #网络结构
    )

    ########
    parser.add_argument(
        '--epoch_plot', default=[200, 1000, 4000]
        #
    )
    ########
    parser.add_argument(
        '--load_path', default='./Data/burger_2.pth'
        #地址
    )
    ########
    parser.add_argument(
        '--is_load', default=False
    )
    parser.add_argument(
        '--N_train', default=10000, type=int
        #训练点的数量(真实解)
    )
    # 训练信息
    parser.add_argument(
        '--epochs', default=60000, type=int
        #总共60000轮
    )
    # parser.add_argument(
    #     '--epochs', default=1, type=int
    # )
    parser.add_argument(
        '--n_f', default=10000, type=int
        #残差点的数量，pde内部随机选点
    )
    parser.add_argument(
        '--n_b_1', default=400, type=int
        #边界条件上x属于[-1,1] t=0均匀分布，选400个点
    )
    parser.add_argument(
        '--n_b_2', default=400, type=int
        #边界条件上t属于[0,1]的均匀分布
    )
    parser.add_argument(
        '--n_data', default=100, type=int
    )
    parser.add_argument(
        '--PDE_panelty', default=1.0, type=float
        #PDE权重
    )
    parser.add_argument(
        '--BC_panelty', default=1.0, type=float
        #边界条件权重
    )
    parser.add_argument(
        '--Data_panelty', default=20.0, type=float
        #精确解或实验值权重
    )
    parser.add_argument(
        '--lr', default=0.001, type=float
        #学习率为0.001
    )
    parser.add_argument(
        '--criterion', default=torch.nn.MSELoss()
        #均方误差
    )
    parser.add_argument(
        '--optimizer', default=torch.optim.Adam
        #adam优化器
    )
    # 网络信息

    parser.add_argument(
        '--activation', default=torch.tanh
        #激活函数为tanh
    )

    parser.add_argument(
        '--t_left', default=0., type=float
        #t左边界，值为0
    )
    parser.add_argument(
        '--t_right', default=1., type=float
        # t右边界，值为1

    )
    parser.add_argument(
        '--x_left', default=-1., type=float
        # X左边界，值为-1
    )
    parser.add_argument(
        '--x_right', default=1., type=float
        # x右边界，值为1
    )
    return parser
