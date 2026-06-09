from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from scipy.ndimage import label


def load_data(*, batch_size, image_size):

    try:

        dataset = load_dataset("train")

        train_dataset = HFDataset(dataset)

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=True
        )

    except Exception as e:
        raise
    while True:
            yield from loader


def _list_image_files_recursively(data_dir):
    images = []
    constraints_pf = []
    loads = []
    deflections = ""
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            images.append(full_path)
        elif "." in entry and ext.lower() in ["npy"]:
            if "load" in entry:
                loads.append(full_path)
            else:
                constraints_pf.append(full_path)
        elif bf.isdir(full_path):
            images.extend(_list_image_files_recursively(full_path))
            loads.extend(_list_image_files_recursively(full_path))
            constraints_pf.extend(_list_image_files_recursively(full_path))
    return images, constraints_pf, loads


def extract_isolated_materials(image, max_material_size=5):


    if image.max() > 1:
        image = image / 255.0
    binary_img = (image > 0.5).astype(np.uint8)


    inverted_img = 1 - binary_img
    labeled_material, num_material = label(inverted_img)

    material_sizes = np.bincount(labeled_material.ravel())


    isolated_materials = np.where((material_sizes > 0) &
                                 (material_sizes < max_material_size))[0]


    isolated_mask = np.zeros_like(binary_img)
    small_materials = np.isin(labeled_material, isolated_materials)
    isolated_mask[small_materials] = 1


    isolated_mask = 1 - isolated_mask
    isolated_mask = isolated_mask * 255.0

    return isolated_mask

def plot_heatmap(matrix, title="Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='viridis', annot=False, cbar=True)
    plt.title(title)
    plt.show()

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

class HFDataset(Dataset):
    def __init__(self, dataset):


        self.dataset = dataset['train']
        self.resolution = 64

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        try:

            sample = self.dataset[idx]


            image = sample['image']
            image = center_crop_arr(image, self.resolution)
            psl_image = sample['psl']
            psl_image = psl_image.convert("RGB")
            psl = center_crop_arr(psl_image, self.resolution)


            image = image.astype(np.float32) / 127.5 - 1
            image = image.reshape(self.resolution, self.resolution, 1)


            psl = np.array(psl).astype(np.float32) / 127.5 - 1
            psl_mask = psl.reshape(self.resolution, self.resolution, 3)


            constraints_pf = np.array(sample['pf'])
            loads = np.array(sample['load'])
            energy_mask = constraints_pf[:, :, 1:2]
            disp_mask = np.array(sample['disp'])
            load_pos = sample['summary']['load_nodes'][0]


            assert constraints_pf.shape[0:2] == image.shape[0:2], "The constraints do not fit the dimension of the image"
            assert loads.shape[0:2] == image.shape[0:2], "The loads do not fit the dimension of the image"
            assert energy_mask.shape[0:2] == image.shape[0:2]
            assert psl_mask.shape[0:2] == image.shape[0:2]
            assert disp_mask.shape[0:2] == image.shape[0:2]


            constraints = np.concatenate([constraints_pf, loads], axis=2)

            load_pos_matrix = np.zeros((1, self.resolution, self.resolution), dtype=int)


            nodes_per_column = self.resolution + 1


            node_zero_based = load_pos - 1


            col = node_zero_based // nodes_per_column
            row = node_zero_based % nodes_per_column


            x = col
            y = self.resolution - row


            i, j = int(x), int(y)
            if 0 <= i < self.resolution and 0 <= j < self.resolution:
                load_pos_matrix[0, i, j] = 1


            i, j = int(x) - 1, int(y)
            if 0 <= i < self.resolution and 0 <= j < self.resolution:
                load_pos_matrix[0, i, j] = 1


            i, j = int(x) - 1, int(y) - 1
            if 0 <= i < self.resolution and 0 <= j < self.resolution:
                load_pos_matrix[0, i, j] = 1


            i, j = int(x), int(y) - 1
            if 0 <= i < self.resolution and 0 <= j < self.resolution:
                load_pos_matrix[0, i, j] = 1


            image_res = np.transpose(image, [2, 0, 1]).astype(np.float32)
            pf_res = np.transpose(constraints, [2, 0, 1]).astype(np.float32)
            energy_mask_res = np.transpose(energy_mask, [2, 0, 1]).astype(np.float32)
            psl_mask_res = np.transpose(psl_mask, [2, 0, 1]).astype(np.float32)
            disp_mask_res = np.transpose(disp_mask, [2, 0, 1]).astype(np.float32)


            out_dict = {}

            return image_res, pf_res, energy_mask_res, psl_mask_res, disp_mask_res, load_pos_matrix, out_dict

        except Exception as e:

            raise
