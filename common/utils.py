

def update_config(cfg, file):
    cfg.defrost()
    cfg.merge_from_file(file)
    cfg.freeze()
    return cfg
