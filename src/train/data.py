import functools
import io
import os
import random
from typing import Callable, Optional
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


try:
    # for slurm
    from petrel_client.client import Client

    client = Client(enable_mc=True)

except ImportError:
    pass


def mask2bounding_box(mask: np.array) -> tuple[int, int, int, int]:
    y, x, _ = np.nonzero(mask)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    return min_x, min_y, max_x, max_y


def custom_collate_fn(
    batch: list[dict[str, float | list[Image.Image]]],
    processor: Callable,
    multi_view: bool,
) -> dict[str, torch.Tensor]:
    images = processor([example["image"] for example in batch])
    if multi_view:
        elevation = torch.stack([torch.tensor(example["elevation"]) for example in batch])
        return {"elevation": elevation, **images}
    else:
        return images


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        world_rank: int = 0,
        world_size: int = 1,
        num_examples: Optional[int] = None,
        num_frames: int = 21,
    ) -> None:
        super(ImageFolderDataset).__init__()
        if "\n" in dataset_path:
            self.image_folder_path, self.meta_file = dataset_path.split("\n")
            self.meta_file_path = os.path.join(self.image_folder_path, self.meta_file)
        else:
            self.image_folder_path = ""
            self.meta_file = self.meta_file_path =  dataset_path
        self.anno = []
        self.num_frames = num_frames
        with open(self.meta_file_path, "rb") as f:
            anno = pickle.load(f)
        for i in range(len(anno['uid'])):
            uid, elevation = anno['uid'][i], anno['elevation'][i]
            dir_path = os.path.join(self.image_folder_path, uid)
            if os.path.isdir(dir_path) and (len(os.listdir(dir_path))) == 16:
                self.anno.append({'uid': uid, 'elevation': elevation})
        random.shuffle(self.anno)
        num_examples = len(self.anno) if num_examples is None else num_examples
        assert num_examples <= len(self.anno), 'Number of examples greater than dataset size.'
        self.anno = self.anno[:num_examples]
        print('Num Examples:', num_examples)

    def __len__(self) -> int:
        return len(self.anno)

    def __getitem__(self, idx: int) -> dict[str, list[np.array]]:
        while True:
            this_anno = self.anno[idx]
            folder_path = os.path.join(self.image_folder_path, this_anno["uid"])
            image_output = []
            try:
                # constant elevation
                elevation = this_anno["elevation"]
                elevation = [elevation,] * self.num_frames
                image_paths = [f'{str(i).zfill(3)}.png' for i in range(16)]
                roll_idx = random.randint(0, len(image_paths) - 1)
                image_paths = image_paths[roll_idx:] + image_paths[:roll_idx]
                image_paths = image_paths[: self.num_frames]
                elevation = elevation[: self.num_frames]
                for i in image_paths:
                    image_path = os.path.join(folder_path, i)
                    image = Image.open(image_path)
                    image_output.append(np.array(image)[:, :, :3])
                break
            except Exception as e:
                print(f"{e}: {folder_path}")
                idx = random.randint(0, len(self.anno) - 1)
                continue
        return {
            "image": image_output,
            "elevation": elevation,
        }


def get_image_folder_dataset(
    args,
    dataset_path: str,
    world_rank: int = 0,
    world_size: int = 1,
    num_examples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 1,
    drop_last: bool = True,
    processor: Callable = None,
    num_frames: int = 21,
) -> DataLoader:
    dataset = ImageFolderDataset(
        dataset_path=dataset_path,
        world_rank=world_rank,
        world_size=world_size,
        num_examples=num_examples,
        num_frames=num_frames,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=8,
        collate_fn=functools.partial(
            custom_collate_fn,
            processor=processor,
            multi_view=args.multi_view,
        ),
    )
    dataloader.num_batches = len(dataset) // batch_size
    return dataloader


def get_dataloader(
    args,
    dataset_type: str,
    dataset_path: str,
    world_rank: int = 0,
    world_size: int = 1,
    num_examples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 1,
    num_frames: int = 21,
    processor: Callable = None,
) -> DataLoader:
    match dataset_type:
        case "image_folder":
            return get_image_folder_dataset(
                args,
                dataset_path=dataset_path,
                world_rank=world_rank,
                world_size=world_size,
                num_examples=num_examples,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                processor=processor,
                num_frames=num_frames,
            )
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
