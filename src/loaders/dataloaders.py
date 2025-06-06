from common.logging import get_logger


def get_dataloaders(config, mode_or_modes):
    logger = get_logger()
    try:
        ds_name = config['dataset']['name']
    except:
        logger.exception(f'you must determine dataset name!')
        
    # check config for <add_boundary_mask>
    try:
        config['dataset']['add_boundary_mask']
    except KeyError:
        config['dataset']['add_boundary_mask'] = False
    
    # check config for <add_boundary_dist>
    try:
        config['dataset']['add_boundary_dist']
    except KeyError:
        config['dataset']['add_boundary_dist'] = False
        
        
    try:
        get_ds_loader = globals()[f"get_{ds_name.lower()}"]
    except:
        logger.exception(f'<get_{ds_name.lower()}> Dataset not implemented yet!')

    _data = get_ds_loader(config, logger=logger, verbose=True)
    
    if isinstance(mode_or_modes, list):
        dataloaders = []
        for mode in mode_or_modes:
            dataloaders.append(_data[mode.lower()]["loader"])
        return dataloaders
    # elif isinstance(mode_or_modes, str):
    #     return _data[mode_or_modes.lower()]["loader"]
    elif ds_name.lower() == "custom":
        from loaders.custom import get_custom
        data = get_custom(config, logger=logger, verbose=True)