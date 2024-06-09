import os
import sys

sys.path.append("..")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.optim as optim
import torch_geometric.data
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import logging
from tqdm import tqdm
from metrics.loss_function import LossRecL2, LossEdge
from eval import eval

model_save_dir = './saved_models/'
logging.basicConfig(format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s',
                    level=logging.INFO)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_simple_dataset(model: torch.nn.Module,
                         train_dataset: torch_geometric.data.Dataset,
                         eval_dataset=None,
                         saved_name='model.pth',
                         lr=0.001,
                         batch_size=4,
                         epoch_num=8,
                         shuffle=False,
                         num_workers=11):
    """
    单数据集训练
    Args:
        train_dataset:
        eval_dataset:
        lr:
        batch_size:
        epoch_num:
        shuffle:
        num_workers:
        saved_name:

    Returns:

    """
    # 消息提醒组件初始化
    logger = logging.getLogger("TrainMode")
    # 记录组建初始化
    writer = SummaryWriter('./log/train')  # tersorboard监视器
    save_path = os.path.join(model_save_dir, saved_name)
    # 模型初始化
    try:
        model.load_state_dict(torch.load(save_path))
        logger.info('Saved model file found and successfully read')
    except:
        logger.warning(
            'No saved model found, new model file will be created:' + str(save_path))
    model.to(device)
    # 数据初始化
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)
    # 损失函数初始化
    loss1 = LossRecL2()
    loss2 = LossEdge()
    loss1.to(device)
    loss2.to(device)
    # 优化器初始化
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    # 主循环
    logger.info('Start of training......')
    epoch_bar = tqdm(range(epoch_num))
    epoch_bar.set_description_str('Training')
    for epoch in epoch_bar:
        # 训练阶段
        loss_mean = 0
        with tqdm(total=len(train_dataloader.dataset)) as batch_bar:
            batch_bar.set_description_str('Epoch %d' % epoch)
            for id, po, gt in train_dataloader:
                # 数据准备
                v_id = id.v.to(device)
                e_id = id.edge_index.to(device)
                v_po = po.v.to(device)
                e_po = po.edge_index.to(device)
                v_gt = gt.v.to(device)
                e_gt = gt.edge_index.to(device)
                v_id = v_id.view(batch_size, -1, 3).permute(0, 2, 1)
                v_po = v_po.view(batch_size, -1, 3).permute(0, 2, 1)
                # 模型预测
                v_pred = model(v_id, v_po)  # B, 3, N
                v_pred = v_pred.permute(0, 2, 1).reshape(-1, 3)
                v_id = v_id.permute(0, 2, 1).reshape(-1, 3)
                # 损失函数
                loss = loss1(v_pred, v_gt) + 5e-4 * loss2(v_pred, e_id, v_id, e_id)
                loss_mean += loss
                # 反向传播更新权重
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # 打印信息
                batch_bar.write(f"batch: {batch_bar.n // train_dataloader.batch_size} loss: {loss:>7f}")
                batch_bar.set_postfix(loss=float(loss), lr=float(optimizer.param_groups[0]['lr']))
                batch_bar.update(train_dataloader.batch_size)
                writer.add_scalar('iteration_loss', loss,
                                  epoch * (len(train_dataloader.dataset) // batch_size) + (batch_bar.n // batch_size))
            batch_bar.reset()
            torch.save(model.state_dict(), save_path)
            loss_mean = loss_mean / (len(train_dataloader.dataset) // batch_size)
            scheduler.step()  # 检测是否自动学习率衰减
            writer.add_scalar('epoch_mean_loss', loss_mean, (epoch + 1) * (len(train_dataloader.dataset) // batch_size))
            # 评估阶段
            if eval_dataset is not None:
                with torch.no_grad():
                    eval_mean = eval(eval_dataset, model=model)
                    writer.add_scalar('epoch_mean_eval', eval_mean,
                                      (epoch + 1) * (len(train_dataloader.dataset) // batch_size))
        logger.info('Model saved to:' + str(save_path))
    logger.info('Train completed!')
    return model

if __name__ == '__main__':
    from DSFFNet.pt_module import FFAdaINModel
    from data_load.npt import NPTDataset

    SMPL_train = NPTDataset(path='../../datasets/smpl-data/train',  # 数据集路径
                            identity_num=16,                        # 身份个数
                            pose_num=800,                           # 姿态个数
                            shuffle_points=True,                    # 是否打乱顶点
                            type='obj')                             # 文件类型obj，可换为ply
    train_simple_dataset(model=FFAdaINModel(),                      # 模型类，必须
                         train_dataset=SMPL_train,                  # 训练数据集，必须
                         # eval_dataset=None,                       # 评估数据集，可选，每个epoch结束后评估一次
                         saved_name='model_smpl.pth',               # 预训练模型名，可选
                         lr=0.0001,                                 # 学习率，必须
                         batch_size=2,                              # 批大小，必须
                         epoch_num=8)                               # epoch数，必须
