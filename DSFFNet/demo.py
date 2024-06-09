import os
import sys
sys.path.append("..")

import torch
import logging
from metrics.eval_function import EvalPMD
from torch.utils.tensorboard import SummaryWriter
from data_load.common import Mesh
import trimesh
import time
model_save_dir = './saved_models/'
logging.basicConfig(format='%(asctime)s - [%(name)s] - %(levelname)s: %(message)s',
                    level=logging.INFO)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def demo(path_id,
         path_po,
         model:torch.nn.Module,
         model_name=None,
         path_gt=None,
         save_name=None,
         tensorboard=False):
    # 消息提示
    logger = logging.getLogger("demo")
    writer = SummaryWriter('./log/demo')  # TensorBoard监视器
    if model_name is not None and isinstance(model_name, str):
        model_path = os.path.join(model_save_dir, model_name)
        try:
            model.load_state_dict(torch.load(model_path))
            logger.info('Saved model file found and successfully read')
        except:
            logger.warning(
                'No saved model found, new model file will be created:' + str(model_path))
    model.to(device)
    model.eval()
    id = Mesh(path_id)
    po = Mesh(path_po)
    # po.add_noise(noise=0.05)
    v_id = id.vertices.t().unsqueeze(0).to(device)
    v_po = po.vertices.t().unsqueeze(0).to(device)
    s_t = time.time()
    v_pred = model(v_id, v_po)
    print("Time: %.3f s" % (time.time() - s_t))
    if tensorboard:
        writer.add_mesh('id_mesh',
                        vertices=v_id[0].t().unsqueeze(0),
                        faces=id.face_index.unsqueeze(0),
                        global_step=0)
        writer.add_mesh('po_mesh',
                        vertices=v_po[0].t().unsqueeze(0),
                        faces=po.face_index.unsqueeze(0),
                        global_step=0)
        writer.add_mesh('pred_mesh',
                        vertices=v_pred[0].t().unsqueeze(0),
                        faces=id.face_index.unsqueeze(0),
                        global_step=0)
    if path_gt is not None:
        gt = Mesh(path_gt)
        v_gt = gt.vertices.t().unsqueeze(0).to(device)
        writer.add_mesh('gt_mesh',
                        vertices=v_gt[0].t().unsqueeze(0),
                        faces=gt.face_index.unsqueeze(0),
                        global_step=0)
        Eval1 = EvalPMD()
        Eval1.to(device)
        eval = Eval1(v_pred, v_gt)
        logger.info('eval: %.6f' % float(eval))
        gt.save_obj('./result/gt.obj')
    v_out = v_pred[0].t().cpu().detach()
    mesh_pred = trimesh.Trimesh(vertices=v_out.numpy(), faces=id.face_index)
    if save_name is not None:
        mesh_pred.export(save_name)
        id.save_obj('./result/id.obj')
        po.save_obj('./result/po.obj')
    logger.info('finished')
    return v_out, mesh_pred


if __name__ == '__main__':
    from DSFFNet.pt_module import FFAdaINModel

    path_id = '../../datasets/smpl-data/id27_600.obj'
    path_po = '../../datasets/smpl-data/id26_500.obj'
    path_gt = '../../datasets/smpl-data/id27_500.obj'

    v_out, mesh_pred = demo(path_id,                            # 目标网格（提供身份），必须
                            path_po,                            # 源网格（提供姿态），必须
                            path_gt=path_gt,                    # 基准网格（用于评估精度），可选，无基准网格则注释该行
                            model=FFAdaINModel(),               # 模型对象，必须
                            model_name='model_smpl.pth',        # 预训练模型名，必须
                            # model_name='model_smal.pth',
                            save_name='./result/out.obj')       # 输出网格名称，必须
