import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
import pathlib
from data_load.common import Mesh
import random
import re
from natsort import natsorted


class NPTDataset(Dataset):
    '''
        Init:
            path: String NPT数据集路径
        Input:
            item: int 文件索引
        Output:
            v_id: Tensor[BVi, 3] identity的坐标,
            n_id: Tensor[BVi, 3] identity的法向量,
            face_index_id: Tensor[F, 3] identity的面索引
            edge_index_id: Tensor[2, Ei] identity的图边索引,
            v_po: Tensor[BVi, 3] pose的坐标,
            n_po: Tensor[BVi, 3] pose的法向量,
            face_index_po: Tensor[F, 3] pose的面索引
            edge_index_po: Tensor[2, Ei] pose的图边索引,
            v_gt: Tensor[BVi, 3] gt的坐标,
            n_gt: Tensor[BVi, 3] gt的法向量,
            face_index_gt: Tensor[F, 3] gt的面索引
            edge_index_gt: Tensor[2, Ei] gt的图边索引,
    '''

    def __init__(self,
                 path='../../datasets/smpl-data/train',
                 identity_num=27,
                 pose_num=800,
                 type='.obj',
                 mode='supervise',
                 shuffle_points=False,
                 filp_yz_axis=False,
                 vertices_normalized=True):
        super(NPTDataset, self).__init__()
        self.data_dir = pathlib.Path(path)
        self.identity_num = identity_num
        self.pose_num = pose_num
        self.shuffle_points = shuffle_points
        self.filp_yz_axis = filp_yz_axis
        self.vertices_normalized = vertices_normalized
        f_list = [str(f) for f in list(self.data_dir.glob('*' + type))]
        f_list = natsorted(f_list)  # 自然排序
        if len(f_list) == 0:
            raise Exception('The destination folder is empty or does not exist: ' + os.path.abspath(path))
        self.file_list = []
        for i in range(identity_num):
            temp = []
            for j in range(pose_num):
                temp.append(f_list[i * pose_num + j])
            self.file_list.append(temp)
        self.id_idx = range(0, self.identity_num)
        self.po_idx = range(0, self.pose_num)
        if mode == 'supervise':
            self.get_func = self.get_supervise
        elif mode == 'unsupervise':
            self.get_func = self.get_unsupervise
        elif mode == 'single':
            self.get_func = self.get_single
    def __len__(self):
        return self.identity_num * self.pose_num

    def __getitem__(self, item):
        return self.get_func()
    def get_supervise(self):
        # id_id = item // self.pose_num
        # id_po = item % self.pose_num
        id_code = np.random.choice(self.id_idx, size=[2], replace=False)
        po_code = np.random.choice(self.po_idx, size=[2], replace=False)
        id_id = id_code[0]
        id_po = po_code[0]
        po_id = id_code[1]
        po_po = id_code[1]
        gt_id = id_code[0]
        gt_po = po_code[1]
        id = Mesh(self.file_list[id_id][id_po], self.vertices_normalized)
        po = Mesh(self.file_list[po_id][po_po], self.vertices_normalized)
        gt = Mesh(self.file_list[gt_id][gt_po], self.vertices_normalized)
        if self.filp_yz_axis:
            id.filp_yz_axis()
            po.filp_yz_axis()
            gt.filp_yz_axis()
        if self.shuffle_points:
            random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                replace=False)  # 随机采样索引，采点数6890
            random_sample_po = np.random.choice(po.vertices.shape[0], size=po.vertices.shape[0],
                                                replace=False)  # 随机采样索引，采点数6890
            id.shuffle_points(random_sample_id)
            po.shuffle_points(random_sample_po)
            gt.shuffle_points(random_sample_id)
        # return id.vertices, po.vertices, gt.vertices
        return Data(v=id.vertices,
                    n=id.normals,
                    face_index=id.face_index.t(),
                    edge_index=id.edge_index,  # 变量名带"index"字样会自动偏置
                    num_nodes=id.vertices.shape[0]), \
            Data(v=po.vertices,
                 n=po.normals,
                 face_index=po.face_index.t(),
                 edge_index=po.edge_index,
                 num_nodes=po.vertices.shape[0]), \
            Data(v=gt.vertices,
                 n=gt.normals,
                 face_index=gt.face_index.t(),
                 edge_index=gt.edge_index,
                 num_nodes=gt.vertices.shape[0])
    def get_unsupervise(self):
        id_code = np.random.choice(self.id_idx, size=[2], replace=False)
        po_code = np.random.choice(self.po_idx, size=[3], replace=False)
        id1_po1 = Mesh(self.file_list[id_code[0]][po_code[0]], self.vertices_normalized)
        id1_po2 = Mesh(self.file_list[id_code[0]][po_code[1]], self.vertices_normalized)
        id2_po3 = Mesh(self.file_list[id_code[1]][po_code[2]], self.vertices_normalized)
        if self.filp_yz_axis:
            id1_po1.filp_yz_axis()
            id1_po2.filp_yz_axis()
            id2_po3.filp_yz_axis()
        # return id.vertices, po.vertices, gt.vertices
        return Data(v=id1_po1.vertices,
                    n=id1_po1.normals,
                    face_index=id1_po1.face_index.t(),
                    edge_index=id1_po1.edge_index,  # 变量名带"index"字样会自动偏置
                    num_nodes=id1_po1.vertices.shape[0]), \
            Data(v=id1_po2.vertices,
                 n=id1_po2.normals,
                 face_index=id1_po2.face_index.t(),
                 edge_index=id1_po2.edge_index,
                 num_nodes=id1_po2.vertices.shape[0]), \
            Data(v=id2_po3.vertices,
                 n=id2_po3.normals,
                 face_index=id2_po3.face_index.t(),
                 edge_index=id2_po3.edge_index,
                 num_nodes=id2_po3.vertices.shape[0])

    def get_single(self):
        id_code = np.random.choice(self.id_idx, size=[1], replace=False)
        po_code = np.random.choice(self.po_idx, size=[1], replace=False)
        id_id = id_code[0]
        id_po = po_code[0]
        id = Mesh(self.file_list[id_id][id_po], self.vertices_normalized)
        if self.filp_yz_axis:
            id.filp_yz_axis()
        if self.shuffle_points:
            random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                replace=False)  # 随机采样索引，采点数6890
            id.shuffle_points(random_sample_id)
        return Data(v=id.vertices,
                    n=id.normals,
                    face_index=id.face_index.t(),
                    edge_index=id.edge_index,  # 变量名带"index"字样会自动偏置
                    num_nodes=id.vertices.shape[0])

class MGDataset(Dataset):
    '''
        Init:
            path: String NPT数据集路径
        Input:
            item: int 文件索引
        Output:
            v_id: Tensor[BVi, 3] identity的坐标,
            n_id: Tensor[BVi, 3] identity的法向量,
            face_index_id: Tensor[F, 3] identity的面索引
            edge_index_id: Tensor[2, Ei] identity的图边索引,
            v_po: Tensor[BVi, 3] pose的坐标,
            n_po: Tensor[BVi, 3] pose的法向量,
            face_index_po: Tensor[F, 3] pose的面索引
            edge_index_po: Tensor[2, Ei] pose的图边索引,
            v_gt: Tensor[BVi, 3] gt的坐标,
            n_gt: Tensor[BVi, 3] gt的法向量,
            face_index_gt: Tensor[F, 3] gt的面索引
            edge_index_gt: Tensor[2, Ei] gt的图边索引,
    '''

    def __init__(self,
                 path='../../datasets/Multi-Garment_dataset/',
                 mesh_num=97,
                 type='.obj',
                 out='id_po',
                 shuffle_points=False,
                 filp_yz_axis=False,
                 vertices_normalized=True):
        super(MGDataset, self).__init__()
        self.data_dir = pathlib.Path(path)
        self.mesh_num = mesh_num
        self.shuffle_points = shuffle_points
        self.filp_yz_axis = filp_yz_axis
        self.vertices_normalized = vertices_normalized
        self.out = out
        f_list = [str(f) for f in list(self.data_dir.glob('*'))]  # 遍历文件夹
        f_list = natsorted(f_list)  # 自然排序
        if len(f_list) == 0:
            raise Exception('The destination folder is empty or does not exist: ' + os.path.abspath(path))
        self.file_list = [f + '/smpl_registered' + type for f in f_list]

    def __len__(self):
        return self.mesh_num

    def __getitem__(self, item):
        if self.out == 'id_po':
            # id_id = item // self.pose_num
            # id_po = item % self.pose_num
            id = random.randint(0, self.mesh_num - 1)
            po = random.randint(0, self.mesh_num - 1)
            po = po if po != id else random.randint(0, self.mesh_num - 1)  # 确保不是同一个
            id = Mesh(self.file_list[id], self.vertices_normalized)
            po = Mesh(self.file_list[po], self.vertices_normalized)
            if self.filp_yz_axis:
                id.filp_yz_axis()
                po.filp_yz_axis()
            if self.shuffle_points:  # 打乱点集
                random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                    replace=False)  # 随机采样索引，采点数6890
                random_sample_po = np.random.choice(po.vertices.shape[0], size=po.vertices.shape[0],
                                                    replace=False)  # 随机采样索引，采点数6890
                id.shuffle_points(random_sample_id)
                po.shuffle_points(random_sample_po)
            # return id.vertices, po.vertices, gt.vertices
            return (Data(v=id.vertices,
                         n=id.normals,
                         face_index=id.face_index.t(),
                         edge_index=id.edge_index,  # 变量名带"index"字样会自动偏置
                         num_nodes=id.vertices.shape[0]),
                    Data(v=po.vertices,
                         n=po.normals,
                         face_index=po.face_index.t(),
                         edge_index=po.edge_index,
                         num_nodes=po.vertices.shape[0]))
        elif self.out == 'id':
            id = random.randint(0, self.mesh_num - 1)
            id = Mesh(self.file_list[id], self.vertices_normalized)
            if self.filp_yz_axis:
                id.filp_yz_axis()
            if self.shuffle_points:  # 打乱点集
                random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                    replace=False)  # 随机采样索引，采点数6890
                id.shuffle_points(random_sample_id)
            # return id.vertices, po.vertices, gt.vertices
            return Data(v=id.vertices,
                        n=id.normals,
                        face_index=id.face_index.t(),
                        edge_index=id.edge_index,  # 变量名带"index"字样会自动偏置
                        num_nodes=id.vertices.shape[0])


class FAUSTDataset(Dataset):
    '''
        Init:
            path: String NPT数据集路径
        Input:
            item: int 文件索引
        Output:
            v_id: Tensor[BVi, 3] identity的坐标,
            n_id: Tensor[BVi, 3] identity的法向量,
            face_index_id: Tensor[F, 3] identity的面索引
            edge_index_id: Tensor[2, Ei] identity的图边索引,
            v_po: Tensor[BVi, 3] pose的坐标,
            n_po: Tensor[BVi, 3] pose的法向量,
            face_index_po: Tensor[F, 3] pose的面索引
            edge_index_po: Tensor[2, Ei] pose的图边索引,
            v_gt: Tensor[BVi, 3] gt的坐标,
            n_gt: Tensor[BVi, 3] gt的法向量,
            face_index_gt: Tensor[F, 3] gt的面索引
            edge_index_gt: Tensor[2, Ei] gt的图边索引,
    '''

    def __init__(self,
                 path='../../datasets/MPI-FAUST/training/registrations/',
                 identity_num=10,
                 pose_num=10,
                 type='.ply',
                 mode='supervise',
                 shuffle_points=False,
                 filp_yz_axis=False,
                 vertices_normalized=True):
        super(FAUSTDataset, self).__init__()
        self.data_dir = pathlib.Path(path)
        self.identity_num = identity_num
        self.pose_num = pose_num
        self.shuffle_points = shuffle_points
        self.filp_yz_axis = filp_yz_axis
        self.vertices_normalized = vertices_normalized
        f_list = [str(f) for f in list(self.data_dir.glob('*' + type))]
        f_list = natsorted(f_list)  # 自然排序
        if len(f_list) == 0:
            raise Exception('The destination folder is empty or does not exist: ' + os.path.abspath(path))
        self.file_list = []
        for i in range(identity_num):
            temp = []
            for j in range(pose_num):
                temp.append(f_list[i * pose_num + j])
            self.file_list.append(temp)
        self.id_idx = range(0, self.identity_num)
        self.po_idx = range(0, self.pose_num)
        if mode == 'supervise':
            self.get_func = self.get_supervise
        elif mode == 'unsupervise':
            self.get_func = self.get_unsupervise
        elif mode == 'single':
            self.get_func = self.get_single

    def __len__(self):
        return 12800

    def __getitem__(self, item):
        return self.get_func()

    def get_supervise(self):
        # id_id = item // self.pose_num
        # id_po = item % self.pose_num
        id_code = np.random.choice(self.id_idx, size=[2], replace=False)
        po_code = np.random.choice(self.po_idx, size=[2], replace=False)
        id_id = id_code[0]
        id_po = po_code[0]
        po_id = id_code[1]
        po_po = id_code[1]
        gt_id = id_code[0]
        gt_po = po_code[1]
        id = Mesh(self.file_list[id_id][id_po], self.vertices_normalized)
        po = Mesh(self.file_list[po_id][po_po], self.vertices_normalized)
        gt = Mesh(self.file_list[gt_id][gt_po], self.vertices_normalized)
        if self.filp_yz_axis:
            id.filp_yz_axis()
            po.filp_yz_axis()
            gt.filp_yz_axis()
        if self.shuffle_points:
            random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                replace=False)  # 随机采样索引，采点数6890
            random_sample_po = np.random.choice(po.vertices.shape[0], size=po.vertices.shape[0],
                                                replace=False)  # 随机采样索引，采点数6890
            id.shuffle_points(random_sample_id)
            po.shuffle_points(random_sample_po)
            gt.shuffle_points(random_sample_id)
        # return id.vertices, po.vertices, gt.vertices
        return Data(v=id.vertices,
                    n=id.normals,
                    face_index=id.face_index.t(),
                    edge_index=id.edge_index,  # 变量名带"index"字样会自动偏置
                    num_nodes=id.vertices.shape[0]), \
            Data(v=po.vertices,
                 n=po.normals,
                 face_index=po.face_index.t(),
                 edge_index=po.edge_index,
                 num_nodes=po.vertices.shape[0]), \
            Data(v=gt.vertices,
                 n=gt.normals,
                 face_index=gt.face_index.t(),
                 edge_index=gt.edge_index,
                 num_nodes=gt.vertices.shape[0])

    def get_unsupervise(self):
        id_code = np.random.choice(self.id_idx, size=[2], replace=False)
        po_code = np.random.choice(self.po_idx, size=[3], replace=False)
        id1_po1 = Mesh(self.file_list[id_code[0]][po_code[0]], self.vertices_normalized)
        id1_po2 = Mesh(self.file_list[id_code[0]][po_code[1]], self.vertices_normalized)
        id2_po3 = Mesh(self.file_list[id_code[1]][po_code[2]], self.vertices_normalized)
        if self.filp_yz_axis:
            id1_po1.filp_yz_axis()
            id1_po2.filp_yz_axis()
            id2_po3.filp_yz_axis()
        # return id.vertices, po.vertices, gt.vertices
        return Data(v=id1_po1.vertices,
                    n=id1_po1.normals,
                    face_index=id1_po1.face_index.t(),
                    edge_index=id1_po1.edge_index,  # 变量名带"index"字样会自动偏置
                    num_nodes=id1_po1.vertices.shape[0]), \
            Data(v=id1_po2.vertices,
                 n=id1_po2.normals,
                 face_index=id1_po2.face_index.t(),
                 edge_index=id1_po2.edge_index,
                 num_nodes=id1_po2.vertices.shape[0]), \
            Data(v=id2_po3.vertices,
                 n=id2_po3.normals,
                 face_index=id2_po3.face_index.t(),
                 edge_index=id2_po3.edge_index,
                 num_nodes=id2_po3.vertices.shape[0])

    def get_single(self):
        id_code = np.random.choice(self.id_idx, size=[1], replace=False)
        po_code = np.random.choice(self.po_idx, size=[1], replace=False)
        id_id = id_code[0]
        id_po = po_code[0]
        id = Mesh(self.file_list[id_id][id_po], self.vertices_normalized)
        if self.filp_yz_axis:
            id.filp_yz_axis()
        if self.shuffle_points:
            random_sample_id = np.random.choice(id.vertices.shape[0], size=id.vertices.shape[0],
                                                replace=False)  # 随机采样索引，采点数6890
            id.shuffle_points(random_sample_id)
        return Data(v=id.vertices,
                    n=id.normals,
                    face_index=id.face_index.t(),
                    edge_index=id.edge_index,  # 变量名带"index"字样会自动偏置
                    num_nodes=id.vertices.shape[0])


class MixedDataset(Dataset):
    def __init__(self,
                 shuffle_points=True,
                 filp_yz_axis=False,
                 vertices_normalized=True,
                 mode='supervise'
                 ):
        super(MixedDataset, self).__init__()
        self.shuffle_points = shuffle_points
        self.filp_yz_axis = filp_yz_axis
        self.vertices_normalized = vertices_normalized
        self.dataset1 = NPTDataset(path='../../datasets/smpl-data/train',
                                   identity_num=16,
                                   pose_num=800,
                                   shuffle_points=shuffle_points,
                                   type='.obj',
                                   mode=mode,
                                   vertices_normalized=vertices_normalized)
        self.dataset2 = FAUSTDataset(path='../../datasets/MPI-FAUST/training/registrations/',
                                     identity_num=10,
                                     pose_num=10,
                                     shuffle_points=shuffle_points,
                                     type='.ply',
                                     mode=mode,
                                     vertices_normalized=vertices_normalized)
        if mode == 'supervise':
            self.get_func = self.get_supervise
        elif mode == 'unsupervise':
            self.get_func = self.get_unsupervise

    def __len__(self):
        return 12800

    def __getitem__(self, item):
        return self.get_func()

    def get_supervise(self):
        '''

        Returns:
            输出id,po,gt
        '''
        i = random.randint(0, 7)
        if i < 4:
            return self.dataset1.__getitem__(0)
        elif i >= 4:
            return self.dataset2.__getitem__(0)

    def get_unsupervise(self):
        '''

        Returns:
            输出id,po,gt
        '''
        i = random.randint(0, 7)
        m1_1, m1_2, m1_3 = self.dataset1.__getitem__(0)
        m2_1, m2_2, m2_3 = self.dataset2.__getitem__(0)
        if i < 4:
            return m1_1, m1_2, m2_3
        elif i >= 4:
            return m2_1, m2_2, m1_3

