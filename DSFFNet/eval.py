import sys
sys.path.append("..")

import torch
from data_load.npt import NPTDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
from metrics.eval_function import EvalPMD


model_save_dir = './saved_models/'
logging.basicConfig(format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s',
                    level=logging.INFO)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def eval(eval_dataset,
         model: torch.nn.Module,
         model_name=None,
         eval_method=EvalPMD(),
         batch_size=8,
         add_mesh=False):
    # 消息提示
    logger = logging.getLogger("EvalMode")
    # 数据收集
    writer = SummaryWriter('./log/eval')  # TensorBoard监视器
    # 模型初始化
    if model_name is not None and isinstance(model_name, str):
        save_path = os.path.join(model_save_dir, model_name)
        try:
            model.load_state_dict(torch.load(save_path))
            logger.info('Saved model file found and successfully read')
        except:
            logger.warning(
                'No saved model found, new model file will be created:' + str(save_path))
    model.to(device)
    model.eval()
    dataloader = DataLoader(eval_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=11,
                            pin_memory=True,
                            drop_last=False)
    Eval1 = eval_method
    Eval1.to(device)
    logger.info('Start of evaluating......')
    # 开始评估
    eval_mean = 0
    with tqdm(total=len(dataloader.dataset)) as eval_bar:
        eval_bar.set_description_str('Evaluating')
        for id, po, gt in dataloader:
            # 数据准备
            v_id = id.v.to(device)
            e_id = id.edge_index.to(device)
            b_id = id.batch.to(device)
            v_po = po.v.to(device)
            e_po = po.edge_index.to(device)
            b_po = po.batch.to(device)
            v_gt = gt.v.to(device)
            with torch.no_grad():
                # 模型预测
                v_id = v_id.view(b_id.max()+1, -1, 3).permute(0, 2, 1)
                v_po = v_po.view(b_po.max()+1, -1, 3).permute(0, 2, 1)
                v_gt = v_gt.view(b_po.max()+1, -1, 3).permute(0, 2, 1)
                v_pred = model(v_id, v_po)
                # 评估函数
                eval = Eval1(v_pred, v_gt)
                eval_mean += eval
            # 打印信息
            eval_bar.write(f"EvalData: {eval_bar.n // dataloader.batch_size} Eval: {eval:>7f}")
            eval_bar.set_postfix(pmd=float(eval))
            writer.add_scalar('eval', eval, eval_bar.n)
            if add_mesh and eval_bar.n % 100 == 0:
                writer.add_mesh('id_mesh',
                                vertices=v_id[0].t().unsqueeze(0),
                                faces=id.face_index.t().unsqueeze(0),
                                global_step=eval_bar.n)
                writer.add_mesh('po_mesh',
                                vertices=v_po[0].t().unsqueeze(0),
                                faces=po.face_index.t().unsqueeze(0),
                                global_step=eval_bar.n)
                writer.add_mesh('pred_mesh',
                                vertices=v_pred[0].t().unsqueeze(0),
                                faces=id.face_index.t().unsqueeze(0),
                                global_step=eval_bar.n)
                writer.add_mesh('gt_mesh',
                                vertices=v_gt[0].t().unsqueeze(0),
                                faces=gt.face_index.t().unsqueeze(0),
                                global_step=eval_bar.n)
            eval_bar.update(dataloader.batch_size)
        eval_bar.reset()
        logger.info('evaluation completed')
        eval_mean = eval_mean/len(dataloader)
    model.train()
    return eval_mean

if __name__ == '__main__':
    from DSFFNet.pt_module import FFAdaINModel
    import os

    SMPL_eval = NPTDataset(path='../../datasets/smpl-data/eval',# 评估集路径
                           identity_num=14,                     # 身份数
                           pose_num=800)                        # 姿态数
    eval_mean = eval(eval_dataset=SMPL_eval,                    # 评估集，必须
                     model=FFAdaINModel(),                      # 模型对象，必须
                     eval_method=EvalPMD(),                     # 评估方法，必须
                     model_name='model_smpl.pth')               # 预训练权重文件，必须
    print("eval_mean: %f" % eval_mean)