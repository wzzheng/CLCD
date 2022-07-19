import os
import torch
import numpy as np
from collections import defaultdict
from functools import partial
from copy import deepcopy
from PIL import Image

import torch.utils.data as data


def get_dict_value(d, keys):
    v = d
    for k in keys:
        v = v[k]
    return v

def set_dict_value(d, keys, value):
    v = d
    for k in keys[:-1]:
        v = v[k]
    v[keys[-1]] = value
    

class DyMLDataset(data.Dataset):
    def __init__(self, dataset_path, m_per_level, drop_extra=False, transform=None, target_transform=None):
        """
        Args:
            dataset_path (str): path to the DyML dataset
            m_per_level (list[int]): [description].
            transform ([type], optional): [description]. Defaults to None.
            target_transform (Callable, optional): Defaults to None.
        """
        self.dataset_path = os.path.join(dataset_path, 'train')
        assert isinstance(m_per_level, list)
        self.m_per_level = m_per_level
        self.num_levels = len(m_per_level) - 1
        self.drop_extra = drop_extra if isinstance(drop_extra, list) else [drop_extra for _ in range(self.num_levels)]
        self.transform = transform
        self.target_transform = target_transform

        self.load_config_file()
        self.get_hierachy()
        self.get_grouped_idx()

    def load_config_file(self):
        with open(os.path.join(self.dataset_path, 'label.csv')) as f:
            data = f.read()
        train_data = data.split('\n')
        train_data[0] = train_data[0].split('_id')[-1]
        train_data = [d.split(', ') for d in train_data]    # fname, coarse, middle, fine
        self.datas = []
        for i, data in enumerate(train_data):
            if len(data) != 4:
                continue
            data[0] = os.path.join(self.dataset_path, 'imgs', data[0])
            # data[1:] = [int(label) for label in data[1:]]
            self.datas.append(data)

        self.classes = {}
        class_info = np.array(self.datas)
        class_info[:, 0] = "all"
        for i in range(self.num_levels, -1, -1):
            self.classes[i] = np.unique(class_info[:, :(i+1)], axis=0)
        # for i, data in enumerate(self.datas):
        #     self.datas[i] = [data[0], *[int(data[i+1]) for i in range(self.num_levels)]]
    
    def get_hierachy(self):
        # define container type
        dictlist = partial(list)
        for _ in range(self.num_levels):
            dictlist = partial(defaultdict, dictlist)
        
        # register information
        self.hierachy = dictlist()
        for i, data in enumerate(self.datas):
            v = get_dict_value(self.hierachy, data[1:])
            v.append(i)
            
    def get_grouped_idx(self):
        self.grouped_idx = []

        hierachy = deepcopy(self.hierachy)
        hierachy = {"all": hierachy}
        for level in range(self.num_levels, -1, -1):
            for _, class_info in enumerate(self.classes[level]):
                v = get_dict_value(hierachy, class_info)

                if isinstance(v, list):
                    np.random.shuffle(v)
                    residue = len(v) % self.m_per_level[level]
                    if residue != 0:
                        v = v[:-residue]
                    v = np.array(v).reshape(-1, self.m_per_level[level])
                    set_dict_value(hierachy, class_info, v)
                    
                else:
                    group = []
                    vlist = list(v.values())
                    probs = np.array([len(vv) for vv in vlist])
                    total_cnt = len(vlist)
                    alive_cnt = sum(probs > 0)
                    mper = self.m_per_level[level]
                    while(sum(probs) >= mper and ((not self.drop_extra[level]) or alive_cnt >= mper)):

                        indices = np.random.choice(total_cnt, size=mper, replace=(alive_cnt<mper), p=probs/sum(probs))
                        sub_group = []
                        for idx in indices:
                            sub_group.append(vlist[idx][0])
                            vlist[idx] = vlist[idx][1:]                            
                            probs[idx] = probs[idx] - 1
                            alive_cnt = sum(probs > 0)
                        sub_group = np.concatenate(sub_group)
                        group.append(sub_group)
                    
                    group = np.stack(group, axis=0) if len(group) > 0 else np.array([])
                    set_dict_value(hierachy, class_info, group)

        self.grouped_idx = torch.from_numpy(hierachy["all"])

    def loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        # from each level sample K images
        
        indices = self.grouped_idx[index]
        info_dict = defaultdict(list)
        info_dict["id"] = indices

        for idx in indices:
            path = self.datas[idx][0]
            target = self.datas[idx][1:]
            target = [int(l) for l in target]
            sample = self.loader(path)

            if self.transform is not None:
                img_dict = self.transform(sample)
            else:
                img_dict = {"data": sample, "addition_data": sample}
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            info_dict["data"].append(img_dict["data"])
            info_dict["addition_data"].append(img_dict["addition_data"])
            info_dict["labels"].append(target)
        
        return info_dict

    def __len__(self):
        return len(self.grouped_idx)



def test_Dyml_dataset():
    ds = DyMLDataset('../../datasets/DyML/dyml_animal', [2, 2, 2, 2])
    print(len(ds))
    sample = ds[1]
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    test_Dyml_dataset()