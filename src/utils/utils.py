import logging, os

def configure_logging(gpu, save_path):
    log_format = '%(asctime)s %(message)s'
    log_level = logging.INFO if gpu == 0 else logging.WARN
    logging.getLogger().setLevel(log_level)
    # print(id(logging.getLogger()))
    # console output
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(ch)
    # file output
    if gpu == 0:
        fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
