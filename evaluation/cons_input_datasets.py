import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from hpgdiff import logger
from datasets import load_dataset
from hpgdiff import logger
from PIL import Image


def load_data(
    *, test_level
):

    try:

        dataset = load_dataset("test")

        test_split = f"test_level_{test_level}"
        test_data = dataset[test_split]

        test_dataset = HFTestDataset(test_data)

        loader = DataLoader(
            test_dataset,
            shuffle=False,
            num_workers=1,
            drop_last=True
        )

    except Exception as e:
        logger.info(f"Failed to load dataset: {str(e)}")
        raise
    while True:
            yield from loader

def center_crop_arr(pil_image, image_size):


    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class HFTestDataset(Dataset):
    def __init__(self, dataset):

        self.dataset = dataset
        self.resolution = 64
        logger.log(f"HF test dataset initialized. Dataset size: {len(self.dataset)}")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        try:

            sample = self.dataset[idx]


            raw_pfs = sample['pf']
            raw_loads = sample['load']
            raw_bcs = sample['bc']
            raw_psl = sample['psl']
            raw_disp = sample['disp']


            psl_image = raw_psl.convert("RGB")
            psl = center_crop_arr(psl_image, self.resolution)


            psl = np.array(psl).astype(np.float32) / 127.5 - 1
            psl_mask = psl.reshape(self.resolution, self.resolution, 3)

            if isinstance(raw_pfs, list):
                raw_pfs = np.array(raw_pfs)


            energy_mask = raw_pfs[:, :, 1:2]


            if isinstance(raw_loads, list):
                raw_loads = np.array(raw_loads)
            if isinstance(raw_bcs, list):
                raw_bcs = np.array(raw_bcs)
            if isinstance(raw_disp, list):
                raw_disp = np.array(raw_disp)

            out_dict = {}

            return (np.transpose(raw_pfs, [2, 0, 1]).astype(np.float32),
                    np.transpose(raw_loads, [2, 0, 1]).astype(np.float32),
                    np.transpose(raw_bcs, [2, 0, 1]).astype(np.float32),
                    np.transpose(energy_mask, [2, 0, 1]).astype(np.float32),
                    np.transpose(psl_mask, [2, 0, 1]).astype(np.float32),
                    np.transpose(raw_disp, [2, 0, 1]).astype(np.float32),
                    out_dict)

        except Exception as e:
            logger.info(f"Error processing sample {idx}: {str(e)}")

            raise
