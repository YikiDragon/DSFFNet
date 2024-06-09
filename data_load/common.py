from typing import overload
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import datetime

import trimesh
class Mesh:
    """
    统一Mesh的数据类型为torch.Tensor
    Init:
        (vertice: np.ndarray|torch.Tensor, faces: np.ndarray|torch.Tensor),
        (mesh: o3d.geometry.TriangleMesh|str)
    Variable:
        vertices: 点集
        normals: 法向量
        faces: 面索引
        tpl_edges: 边索引
    """

    @overload
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, vertices_normalized: bool) -> None:
        ...

    @overload
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, vertices_normalized: bool) -> None:
        ...

    @overload
    def __init__(self, mesh: str, vertices_normalized: bool) -> None:
        ...

    @overload
    def __init__(self, mesh: trimesh.Trimesh, vertices_normalized: bool) -> None:
        ...

    def __init__(self, *args):
        if isinstance(args[0], str):
            try:
                self.mesh: trimesh.Trimesh = trimesh.load(args[0])
            except:
                raise Exception('Can not read file: '+args[0])
        elif isinstance(args[0], trimesh.Trimesh):
            self.mesh: trimesh.Trimesh = args[0]
        elif isinstance(args[0], np.ndarray):
            try:
                self.mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=args[0], faces=args[1])
            except:
                raise Exception('Input parameter error')
        elif isinstance(args[0], torch.Tensor):
            try:
                self.mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=args[0].numpy(), faces=args[1].numpy())
            except:
                raise Exception('Input parameter error')
        else:
            raise Exception('Input parameters are not satisfied')
        # self.vertices = torch.Tensor(self.mesh.vertices)  # N,3
        if (len(args) == 2 and not args[1]) or (len(args) == 3 and not args[2]):
            self.normalize = False
            self.vertices = torch.tensor(self.vertices_centered(), dtype=torch.float)
        else:
            self.normalize = True
            self.vertices = torch.tensor(self.vertices_normalized(), dtype=torch.float)  # N,3
        self.normals = torch.tensor(np.copy(self.mesh.vertex_normals), dtype=torch.float)
        self.face_index = torch.tensor(self.mesh.faces)                           # F,3
        self.edge_index = torch.tensor(self.get_tpl_edges_trimesh()).t()          # 2,E
        # self.edge_index = torch.tensor(self.get_limit_adj_vertex()).t()  # 2,E
        # self.edge_index = torch.tensor(self.get_unique_edges()).t()  # 2,E

    def shuffle_points(self, random_sample=None):
        if random_sample is None:
            random_sample = np.random.choice(self.vertices.shape[0], size=self.vertices.shape[0], replace=False)  # 随机采样索引，采点数6890
        new_vertices = self.vertices[random_sample]
        new_normals = self.normals[random_sample]
        face_dict = {}
        for tar_idx, src_idx in enumerate(random_sample):
            face_dict[src_idx] = tar_idx
        # for i in range(len(random_sample)):
        #     face_dict[random_sample[i]] = i
        new_face = []
        for i in range(self.face_index.shape[0]):
            new_face.append([face_dict[int(self.face_index[i][0])],
                             face_dict[int(self.face_index[i][1])],
                             face_dict[int(self.face_index[i][2])]])
        new_face = torch.tensor(new_face)
        # new_face = self.face_index
        # for i in range(self.vertices.shape[0]):
        #     new_face = torch.where(self.face_index == i, np.argwhere(random_sample == i), new_face)
        # update data
        self.vertices = new_vertices
        self.normals = new_normals
        self.face_index = new_face
        # self.face_index = torch.LongTensor(new_face)
        self.mesh: trimesh.Trimesh = trimesh.Trimesh(vertices=self.vertices.numpy(), faces=self.face_index.numpy())
        self.edge_index = torch.tensor(self.get_tpl_edges_trimesh()).t()  # 2,E
        # self.edge_index = torch.tensor(self.get_limit_adj_vertex()).t()  # 2,E
        # self.edge_index = torch.tensor(self.get_unique_edges()).t()  # 2,E

    def get_unique_edges(self):
        return self.mesh.edges_unique

    def get_tpl_edges_trimesh(self):
        """
        获取可用于EdgeConv的边集合
        Returns:
            tpl_edges: [2, E] 包含每个点与其邻点的边索引
        """
        vertex_neighbors = self.mesh.vertex_neighbors
        tpl_edges = []
        i = np.arange(len(vertex_neighbors))
        def func(i):
            v_n = np.expand_dims(np.array(vertex_neighbors[i]), axis=1)
            v_n = np.concatenate((i * np.ones_like(v_n), v_n), axis=1)
            tpl_edges.append(v_n)
        _ = np.vectorize(func)(i)
        tpl_edges = np.concatenate(tpl_edges, axis=0)
        # for i, v_n in enumerate(vertex_neighbors):
        #     v_n = np.expand_dims(np.array(vertex_neighbors[i]), axis=1)
        #     v_n = np.concatenate((i * np.ones_like(v_n), v_n), axis=1)
        #     tpl_edges.append(v_n)
        # tpl_edges = np.concatenate(tpl_edges, axis=0)
        return tpl_edges

    def get_limit_adj_vertex(self, num=2):
        vertex_neighbors = self.mesh.vertex_neighbors
        adj_vertex = np.zeros(shape=(len(vertex_neighbors),2), dtype=np.int64)
        i = np.arange(len(vertex_neighbors))
        def func(i):
            adj_vertex[i] = np.array(vertex_neighbors[i][0:num])
        # def func(v_n):
        #     v_n = np.array(v_n, dtype=np.int32)[0:num]
        _ = np.vectorize(func)(i)
        return adj_vertex

    def vertices_centered(self):
        bbox_min, bbox_max = self.mesh.bounds
        bbox_center = (bbox_min + bbox_max) / 2
        return (self.mesh.vertices - bbox_center)

    def vertices_normalized(self):
        """
        点坐标归一化
        Returns:
            vertices [N,3] 归一化后的点集坐标
        """
        bbox_min, bbox_max = self.mesh.bounds
        bbox_center = (bbox_min + bbox_max)/2
        scale = np.max(bbox_max-bbox_min) / 2
        return (self.mesh.vertices - bbox_center) / scale
        # return (self.mesh.vertices - bbox_min) / scale

    def add_noise(self, noise=0.05):
        noise = np.random.uniform(-noise, noise, self.mesh.vertices.shape)
        self.mesh.vertices = self.mesh.vertices + noise
        if self.normalize:
            self.vertices = torch.tensor(self.vertices_centered(), dtype=torch.float)
        else:
            self.vertices = torch.tensor(self.vertices_normalized(), dtype=torch.float)  # N,3
        self.normals = torch.tensor(np.copy(self.mesh.vertex_normals), dtype=torch.float)
        self.face_index = torch.tensor(self.mesh.faces)                           # F,3
        self.edge_index = torch.tensor(self.get_tpl_edges_trimesh()).t()          # 2,E

    def filp_yz_axis(self):
        vertices = self.mesh.vertices
        vertices[:, 1] = self.mesh.vertices[:, 2]
        vertices[:, 2] = self.mesh.vertices[:, 1]
        self.mesh.vertices = vertices
        if self.normalize:
            self.vertices = torch.tensor(self.vertices_centered(), dtype=torch.float)
        else:
            self.vertices = torch.tensor(self.vertices_normalized(), dtype=torch.float)  # N,3
        self.normals = torch.tensor(np.copy(self.mesh.vertex_normals), dtype=torch.float)
        self.face_index = torch.tensor(self.mesh.faces)                           # F,3
        self.edge_index = torch.tensor(self.get_tpl_edges_trimesh()).t()          # 2,E

    def view(self, save_obj=False, save_path=None):
        """
        展示三维模型
        Args:
            save_obj: 是否保存obj
            save_path: 指定保存路径
        """
        if save_obj:
            if save_path is None:
                save_path = os.path.join('../saved_obj',
                                         'Mesh' + datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S') + '.obj')  # 自定目录
            self.mesh.export(save_path)
            print('The obj has been saved: ' + save_path)
        self.mesh.show()

    def save_obj(self, save_path):
        self.mesh.export(save_path)
        print('The obj has been saved: ' + save_path)
        return True

    def view_snapshot(self, resolution=(480, 640), save_img=False, save_path=None):
        """
        使用plt展示Mesh的快照，保存图像（可选）
        Args:
            resolution: 分辨率
            save_img:  是否保存图像
            save_path: 保存路径
        Returns:
            img: 图像数组
        """
        # 注意要求pyglet版本在1.5.4-1.5.15之间才可正常工作
        scene = trimesh.Scene()
        scene.add_geometry(self.mesh)
        r_e = trimesh.transformations.euler_matrix(0, 0, 0, "ryxz",)        # 不旋转mesh
        t_r = scene.camera.look_at(self.mesh.bounds, rotation=r_e)          # 矫正镜头以包裹
        scene.camera_transform = t_r
        png = scene.save_image(resolution=resolution)
        file = io.BytesIO(png)
        img = plt.imread(file)
        plt.figure()
        plt.imshow(img)
        if save_img:
            if save_path is None:
                save_path = os.path.join('../saved_images',
                                         'Mesh' + datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S') + '.pdf')  # 自定目录
            plt.savefig(save_path)
            print('The image has been saved: '+save_path)
        plt.show()
        return img
