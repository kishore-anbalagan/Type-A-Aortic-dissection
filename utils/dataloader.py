from torch.utils.data import DataLoader

from utils.dataset_loader import ImageTAADDataset

import config


def get_dataloader():

    dataset = ImageTAADDataset(
        config.DATASET_PATH
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return loader