import os
from PIL import Image


class DVR_product_Test():
    def __init__(self, dataset, level='coarse', transform=None):
        assert level in ['coarse', 'middle', 'fine']
        level_to_labelid = {'coarse':1, 'middle':2, 'fine':3}
        self.labelid = level_to_labelid[level]

        self.dataset = dataset
        self.fetcher = None
        self.transform = transform
        self.dataset_path = os.path.join(dataset, 'mini-bmk_all_in_one')
        self.metas = []
        self.load_config_file()

    def load_config_file(self):
        with open(os.path.join(self.dataset_path, 'label.csv')) as f:
            data = f.read()
        train_data = data.split('\n')
        train_data[0] = train_data[0].split('_id')[-1]
        self.datas = [d.split(', ') for d in train_data]    # fname, coarse, middle, fine
        for i, d in enumerate(self.datas):
            if len(d) != 4:
                continue
            # fname, coarse, middle, fine = d
            self.metas.append([d[0], int(d[self.labelid])])

    def loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        meta = self.metas[idx]
        img = self.loader(os.path.join(self.dataset_path, 'imgs', meta[0]))
        if self.transform is not None:
            img_dict = self.transform(img)
        else:
            img_dict = {"data": img, "addition_data": img}
        info_dict = {
            "labels": meta[1],
            "id": idx
        }
        info_dict.update(img_dict)
        return info_dict

    # def __getitem__(self, idx):

    #     meta = self.metas[idx]
    #     img = cv2.imread(os.path.join(self.dataset_path, 'imgs', meta[0]))
    #     if self.transform is not None:
    #         img = self.transform(image=img)["image"]
    #     return {"data": img.astype("uint8"), "label": meta[3], "mid_g_label": meta[2], "low_g_label": meta[1]}

    def __len__(self):
        return len(self.metas)


class DVR_animal_Test():
    def __init__(self, dataset, level, transform=None, mode="query"):
        assert level in ["fine", "middle", "coarse"]
        self.dataset = dataset
        self.fetcher = None
        self.transform = transform
        self.mode = mode

        self.dataset_path = os.path.join(dataset, 'bmk_' + level)
        self.metas = []
        self.load_config_file()

    def load_config_file(self):
        with open(os.path.join(self.dataset_path, self.mode + '.csv')) as f:
            data = f.read()
        train_data = data.split('\n')
        train_data[0] = train_data[0].split('_id')[-1]
        self.datas = [d.split(', ') for d in train_data]    # fname, coarse, middle, fine
        for i, d in enumerate(self.datas):
            if len(d) != 2:
                continue
            # fname, coarse, middle, fine = d
            d[1] = int(d[1])
            self.metas.append(d)

    def loader(self, path: str) -> Image.Image:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        meta = self.metas[idx]
        img = self.loader(os.path.join(self.dataset_path, self.mode, meta[0]))
        if self.transform is not None:
            img_dict = self.transform(img)
        else:
            img_dict = {"data": img, "addition_data": img}
        info_dict = {
            "labels": meta[1],
            "id": idx
        }
        info_dict.update(img_dict)
        return info_dict
            
    def __len__(self):
        return len(self.metas)


def test_Dyml_dataset():
    ds = DVR_animal_Test('../../datasets/DyML/dyml_animal', 'coarse')
    print(len(ds))
    sample = ds[10]
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    test_Dyml_dataset()

