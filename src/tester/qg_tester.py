import torch
import logging
from tqdm import tqdm
import torch.distributed as dist

class QGTester:

    def __init__(
        self,
        batch_size,
        dataset_num_workers=8,
        is_normalize=True,
        splits_to_eval=["test"],
        is_distributed=False,
    ):
        self.batch_size = batch_size
        self.dataset_num_workers = dataset_num_workers
        self.is_normalize = is_normalize
        self.splits_to_eval = splits_to_eval
        self.is_distributed = is_distributed

        self.initiate_property()
    
    """
    Initialization
    """
    
    def initiate_property(self):
        self.trainable_object_list = [
            "models",
            "collectors",
        ]
        
    def initiate_datasets(self):
        # self.datasets = {
        #     k: self.datasets[k] for k in self.splits_to_eval
        # }
        datasets = {}
        for split in self.splits_to_eval:
            datasets[split+'_q'] = self.datasets[split+'_q']
            datasets[split+'_g'] = self.datasets[split+'_g']
        self.datasets = datasets
            
    
    """
    Set and Get
    """

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_distributed(self, flag=True):
        self.is_distributed = flag
    
    """
    test
    """
    def prepare(
        self,
        models,
        datasets,
        evaluators,
        device,
        device_ids,
    ):
        """
        Load objects to be tested.

        Args:
            models (dict):
                Dictionary of models.
            datasets (dict):
                Dictionary of datasets.
            evaluators (dict):
                Dictioanry of evaluators.
            device (device):
                Computation device.
            device_ids (list(int)):
                Instruct Faiss package to use the corresponding devices.
        """
        # pass parameters
        self.models = models
        self.datasets = datasets
        self.evaluators = evaluators
        self.device = device
        self.device_ids = device_ids

    def test(self):
        self.initiate_datasets()

        self.set_to_eval()
        outputs = {}
        with torch.no_grad():
            for split in self.splits_to_eval:
                q_dataset = self.datasets[split+"_q"]
                g_dataset = self.datasets[split+"_g"]
                self.initiate_dataloader(dataset=q_dataset)
                self.q_embeddings, self.q_labels = self.get_embeddings()
                self.initiate_dataloader(dataset=g_dataset)
                self.g_embeddings, self.g_labels = self.get_embeddings()
                results = self.compute_metrics()
                outputs[split] = results
        return outputs
    
    def set_to_eval(self):
        for trainable_name in self.trainable_object_list:
            trainable_object = getattr(self, trainable_name, None)
            if trainable_object  is None:
                logging.warn(
                    "{} is not a member of trainer".format(
                        trainable_name
                    )
                )
            else:
                for v in trainable_object.values():
                    v.eval()

    def initiate_dataloader(self, dataset):
        logging.info(
            "{}: Initiating dataloader".format(
                self.__class__.__name__
            )
        )
        sampler = None
        # get dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(self.batch_size),
            sampler=sampler,
            drop_last=False,
            pin_memory=False,
            shuffle=False,
            num_workers=self.dataset_num_workers
        )
        self.dataloader_iter = iter(self.dataloader)
    
    def get_embeddings(self):
        logging.info(
            "Compute eval embeddings"
        )
        rank = dist.get_rank()
        pbar = tqdm(self.dataloader_iter, disable=(rank!=0))
        embeddings_list, labels_list = [], []
        for info_dict in pbar:
            # print(info_dict["data"].shape)
            # print(info_dict["labels"].shape)
            # get data
            data = info_dict["data"].to(self.device)
            label = info_dict["labels"].to(self.device)
            # forward
            embedding = self.compute_embeddings(data)
            embeddings_list.append(embedding)
            labels_list.append(label)
        embeddings = torch.cat(embeddings_list)
        labels = torch.cat(labels_list)

        # to numpy
        embeddings = embeddings.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        return embeddings, labels


    
    def compute_embeddings(self, data):
        embedding = self.forward_models(data)
        return (
            torch.nn.functional.normalize(embedding, dim=-1) 
            if self.is_normalize 
            else embedding
        )
    
    def forward_models(self, data):
        embedding_trunk = self.models["trunk"](
            data
        )
        embedding_embedder = self.models["embedder"](
            embedding_trunk
        )
        return embedding_embedder
        # return embedding_trunk
    
    def compute_metrics(self):
        metrics_dict = self.evaluators["default"].get_accuracy(
            self.q_embeddings,
            self.g_embeddings,
            self.q_labels,
            self.g_labels,
            True,
            device_ids=self.device_ids
        )
        return metrics_dict
    
