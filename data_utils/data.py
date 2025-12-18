import os
import glob
# import h5py
import numpy as np
from torch.utils.data import Dataset
from .all_tools import read_ply2np
import random
import torch


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        str_download_add='wget %s' % (www)
        str_download_add+=' --no-check-certificate'
        os.system(str_download_add)
        os.system('unzip %s' % (zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc




class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point

class PartPlants(Dataset):
    def __init__(self, root='', npoints=2500, split='train', class_choice=None, normal_channel=False,sample_num=1,loop=1,transform=None):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'classs2number.txt')
        # self.catfile = os.path.join(self.root, 'split_2/classs2number.txt')

        self.cat = {}
        self.normal_channel = normal_channel
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)
        self.meta = {}
        train_list = []
        test_list = []
        val_list = []
        self.cloud_names = []
        with open(os.path.join(self.root, 'split', 'train.txt'), 'r') as f:
            for line in f:
                train_list.append(line.strip().split("/")[1].split(".")[0])
        with open(os.path.join(self.root, 'split', 'val.txt'), 'r') as f:
            for line in f:
                val_list.append(line.strip().split("/")[1].split(".")[0])
        with open(os.path.join(self.root, 'split', 'test.txt'), 'r') as f:
            for line in f:
                test_list.append(line.strip().split("/")[1].split(".")[0])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_list) or (fn[0:-4] in val_list))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_list]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_list]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_list]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.ply'))
                # self.meta[item].append(os.path.join(dir_point, token + '.txt'))

                self.cloud_names.append(token)
        self.loop = loop
        self.transform =transform
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if split == 'test':
            self.datapath = self.datapath
        else:
            if sample_num<len(self.datapath):
                self.datapath = random.sample(self.datapath,sample_num)
            else:
                self.datapath = self.datapath
        self.data_idx = np.arange(len(self.datapath))

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index% len(self.data_idx)]
        else:
            fn = self.datapath[index%len(self.data_idx)]
            cat = self.datapath[index% len(self.data_idx)][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = read_ply2np(fn[1]).astype(np.float32)
            # data = np.loadtxt(fn[1]).astype(np.float32)
            if self.npoints<10000:
                # data = farthest_point_sample(data, self.npoints)
                # resample
                choice = np.random.choice(len(data), self.npoints, replace=True)
                data = data[choice, :]
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.transform is not None:
            point_set[:, 0:3] = self.transform(point_set[:, 0:3])


        return point_set, cls, seg,self.cloud_names[index% len(self.data_idx)]

    def __len__(self):
        return len(self.datapath)* self.loop


class Cabbage_2_cls(Dataset):
    def __init__(self,  split='train',loop=10,transform=None):
        super().__init__()
        self.data_root = "/home/zero/mnt/sda6/split_data_instance/Cabbage/all_cabbage/Cabbage"
        self.split = split
        if self.split == 'train':
            self.data_list =  sorted(os.listdir("/home/zero/mnt/sda6/split_data_instance/Cabbage/all_cabbage/Cabbage/train"))
        else:
            self.data_list = sorted(os.listdir("/home/zero/mnt/sda6/split_data_instance/Cabbage/all_cabbage/Cabbage/test"))
        self.transform = transform
        self.loop = loop
        self.data_idx = np.arange(len(self.data_list))
    def __getitem__(self, idx):
        data_idx = self.data_list[idx % len(self.data_list)]
        # item = self.data_list[data_idx]
        data_foder = os.path.join(self.data_root, self.split)
        data_path = os.path.join(data_foder, data_idx)
        data = np.loadtxt(data_path)
        coord, seg = data[:, 0:3], data[:, -1]
        cls = np.array([0]).astype(np.int32)
        coord = pc_normalize(coord)
        if self.transform is not None:
            coord = self.transform(coord)

        return coord, cls, seg,self.data_list[idx% len(self.data_list)]

    def __len__(self):
        return len(self.data_list) * self.loop


