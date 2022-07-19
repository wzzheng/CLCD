import torch.backends.cudnn as cudnn
import torch.distributed as dist
import logging
import os, sys
from os.path import join as opj
# workspace = os.environ["WORKSPACE"]
os.environ["WORKSPACE"] = "./"
sys.path.insert(0, "./")

from copy import deepcopy
logging.getLogger().setLevel(logging.INFO)

from gedml.launcher.misc import ParserWithConvert
from gedml.launcher.creators import ConfigHandler
from gedml.launcher.misc import utils

from src import collatefn, collector, loss, model, transform, dataset, tester
from src.utils import configure_logging


# argparser
csv_path = os.path.abspath(opj(__file__, "../config/args.csv"))
parser = ParserWithConvert(csv_path=csv_path, name="GeDML")
opt, convert_dict = parser.render()

# args postprocess
opt.save_path = opj(opt.save_path, opt.save_name)
if opt.is_resume:
    opt.delete_old = False

# hyper-parameters
phase = "train"
# phase = "evaluate"
is_test = True
is_save = True


cudnn.deterministic = True
cudnn.benchmark = True

# get confighandler
config_root = os.path.abspath(opj(__file__, "../config/"))
if opt.link_path is None:
    link_root = os.path.join(config_root, "links")
    if opt.setting is None:
        opt.link_path = opj(link_root, "link.yaml")
    else:
        opt.link_path = os.path.join(link_root, "link_" + opt.setting + ".yaml")
opt.assert_path = os.path.join(config_root, "assert.yaml")
opt.param_path = os.path.join(config_root, "param")
opt.wrapper_path = os.path.join(config_root, "wrapper")


def main_worker(rank, opt):
    configure_logging(rank, opt.save_path)

    if opt.is_distributed:
        dist.init_process_group(
            backend="nccl",
            world_size=opt.world_size,
            rank=rank
        )
        dist.barrier()
        if not os.path.exists(opt.save_path):
            os.mkdir(opt.save_path)
        opt.save_path = opj(opt.save_path, "gpu_{}".format(rank))
    
    opt.device = rank
        
    config_handler = ConfigHandler(
        convert_dict=convert_dict,
        link_path=opt.link_path,
        assert_path=opt.assert_path,
        params_path=opt.param_path,
        wrapper_path=opt.wrapper_path,
        is_confirm_first=False
    )

    config_handler.creator_manager.register_packages('collatefns', collatefn)
    config_handler.creator_manager.register_packages('collectors', collector)
    config_handler.creator_manager.register_packages('losses', loss)
    config_handler.creator_manager.register_packages('models', model)
    config_handler.creator_manager.register_packages('transforms', transform)
    config_handler.creator_manager.register_packages('datasets', dataset)
    config_handler.creator_manager.register_packages('testers', tester)

    # initiate params_dict
    modify_link_dict={
        "datasets": [
            {"train": "{}_train.yaml".format(opt.dataset)},
            {"test": "{}_test.yaml".format(opt.dataset)}
        ]
    }
    params_dict = config_handler.get_params_dict()

    # delete redundant options
    opt_dict = deepcopy(opt.__dict__)
    convert_opt_list = list(config_handler.convert_dict.keys())
    for k in list(opt_dict.keys()):
        if k not in convert_opt_list:
            opt_dict.pop(k)

    # modify parameters
    objects_dict = config_handler.create_all(opt_dict)

    # get manager
    manager = utils.get_default(objects_dict, "managers")

    # get recorder
    recorder = utils.get_default(objects_dict, "recorders")

    # start
    manager.run(
        phase=phase,
        start_epoch=opt.start_epoch,
        total_epochs=100,
        is_test=is_test,
        is_save=is_save,
        warm_up=0
    )

if __name__ == "__main__":
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    if opt.is_distributed:
        import torch.multiprocessing as mp
        mp.spawn(main_worker, nprocs=opt.world_size, args=(deepcopy(opt),))
    else:
        main_worker(0, opt)
