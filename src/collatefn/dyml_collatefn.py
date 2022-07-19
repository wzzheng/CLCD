import torch, numpy as np

class DyMLCollateFn(object):
    def __init__(self):
        pass

    def __call__(self, info_dict: list):
        assert isinstance(info_dict, list)
        elem = info_dict[0]
        assert isinstance(elem, dict)
        info_dict = {
            key: self._concatenate_tensor(
                [item[key] for item in info_dict]
            )
            for key in elem
        }
        info_dict = self.modify_info_dict(info_dict)
        return info_dict
    
    def modify_info_dict(self, info_dict: dict) -> dict:
        info_dict.pop("id")
        data_stream2 = info_dict.pop("addition_data")
        info_dict["data"] = torch.cat([info_dict["data"], data_stream2], dim=0)
        return info_dict
    
    def _concatenate_tensor(self, batch: torch.Tensor):
        elem = batch[0]
        out = None
        if isinstance(elem, list):
            if isinstance(elem[0], (torch.Tensor, list)):
                batch = [b for a in batch for b in a]
                elem = batch[0]
        
        if isinstance(elem, torch.Tensor):
            if torch.utils.data.get_worker_info() is not None:
                # modified from: https://github.com/pytorch/pytorch/torch/utils/data/_utils/collate.py
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel) # TODO: What's Tensor.FloatStorage? 
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, np.int64):
            return torch.tensor(batch)
        elif isinstance(elem, list):
            return torch.tensor(batch)
        else:
            raise TypeError("Wrong type {}".format(type(elem)))
