from datasets.custom import CustomDatasetFast
from torch.utils.data import DataLoader
from modules.transforms import DiffusionTransform

def get_custom(config, logger=None, verbose=False):
    if logger: 
        print = logger.info

    INPUT_SIZE = config["dataset"]["input_size"]
    DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))

    # preparing training dataset
    tr_dataset = CustomDatasetFast(
        mode="tr",
        data_dir=config["dataset"]["data_dir"],
        one_hot=False,
        image_size=config["dataset"]["input_size"],
        img_transform=DT.get_forward_transform_img(),
        msk_transform=DT.get_forward_transform_msk(),
        add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
        add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
        logger=logger
    )
    
    vl_dataset = CustomDatasetFast(
        mode="vl",
        data_dir=config["dataset"]["data_dir"],
        one_hot=False,
        image_size=config["dataset"]["input_size"],
        img_transform=DT.get_forward_transform_img(),
        msk_transform=DT.get_forward_transform_msk(),
        add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
        add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
        logger=logger
    )
    
    te_dataset = CustomDatasetFast(
        mode="te",
        data_dir=config["dataset"]["data_dir"],
        one_hot=False,
        image_size=config["dataset"]["input_size"],
        img_transform=DT.get_forward_transform_img(),
        msk_transform=DT.get_forward_transform_msk(),
        add_boundary_mask=config["dataset"].get("add_boundary_mask", False),
        add_boundary_dist=config["dataset"].get("add_boundary_dist", False),
        logger=logger
    )

    if verbose:
        print("Custom Dataset:")
        print(f"├──> Length of training_dataset:   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:       {len(te_dataset)}")

    # prepare dataloaders
    tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])
    vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])
    te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

    return {
        "tr": {"dataset": tr_dataset, "loader": tr_dataloader},
        "vl": {"dataset": vl_dataset, "loader": vl_dataloader},
        "te": {"dataset": te_dataset, "loader": te_dataloader},
    }
