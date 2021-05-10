import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from shutil import copy


class RGBDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(
        self,
        dataset_dir,
        splits_dir,
        output_dir,
        dataset_percentage=1.0,
        split="train",
        clip_len=16,
        preprocess=False,
    ):
        self.dataset_dir, self.splits_dir, self.output_dir = dataset_dir, splits_dir, output_dir
        self.dataset_percentage = dataset_percentage
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You need to download it from official website."
            )

        if (not self.check_preprocess()) or preprocess:
            print(
                "Preprocessing of {} dataset, this will take long, but it will be done only once.".format(
                    dataset_dir
                )
            )
            self.preprocess()


        self.fnames, labels = [], []
        for txt_name in sorted(os.listdir(self.splits_dir)):
            label_name = '_'.join(txt_name.split("_")[:-2])
            f = open(os.path.join(self.splits_dir, txt_name), "r")
            for avi_name in f.readlines():
                self.fnames.append(os.path.join(self.dataset_dir, avi_name[:-7]))
                labels.append(label_name)

        assert len(labels) == len(self.fnames)
        print("Number of {} videos: {:d}".format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {
            label: index for index, label in enumerate(sorted(set(labels)))
        }
        # Convert the list of label names into an array of label indices
        self.label_array = np.array(
            [self.label2index[label] for label in labels], dtype=int
        )

        if not os.path.exists("dataloaders/hmdb_labels.txt"):
            with open("dataloaders/hmdb_labels.txt", "w") as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id + 1) + " " + label + "\n")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == "test":
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.dataset_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, "train")):
            return False
        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, "train"))
            os.mkdir(os.path.join(self.output_dir, "val"))
            os.mkdir(os.path.join(self.output_dir, "test"))

        # Split train/val/test sets
        for file in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file)
            video_files = os.listdir(file_path)

            train_and_valid, test = train_test_split(
                video_files, test_size=0.2, random_state=42
            )
            train, val = train_test_split(
                train_and_valid, test_size=0.2, random_state=42
            )

            if self.dataset_percentage < 1.0:
                drop_percentage = 1 - self.dataset_percentage
                train, _ = train_test_split(
                    train, test_size=drop_percentage, random_state=42
                )
                val, _ = train_test_split(
                    val, test_size=drop_percentage, random_state=42
                )
                test, _ = train_test_split(
                    test, test_size=drop_percentage, random_state=42
                )

            train_dir = os.path.join(self.output_dir, "train")
            val_dir = os.path.join(self.output_dir, "val")
            test_dir = os.path.join(self.output_dir, "test")

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                dir_name = os.path.join(train_dir, file)
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                copy(os.path.join(self.dataset_dir, file, video), dir_name)

            for video in val:
                dir_name = os.path.join(val_dir, file)
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                copy(os.path.join(self.dataset_dir, file, video), dir_name)

            for video in test:
                dir_name = os.path.join(test_dir, file)
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                copy(os.path.join(self.dataset_dir, file, video), dir_name)

        print("Preprocessing finished.")

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty(
            (frame_count, self.resize_height, self.resize_width, 3), np.dtype("float32")
        )
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[
            time_index : time_index + clip_len,
            height_index : height_index + crop_size,
            width_index : width_index + crop_size,
            :,
        ]

        return buffer


class FlowDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(
        self,
        dataset_dir,
        splits_dir,
        output_dir,
        dataset_percentage=1.0,
        split="train",
        clip_len=16,
        preprocess=False,
    ):
        self.dataset_dir, self.splits_dir, self.output_dir = dataset_dir, splits_dir, output_dir
        self.dataset_percentage = dataset_percentage
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You need to download it from official website."
            )

        if (not self.check_preprocess()) or preprocess:
            print(
                "Preprocessing of {} dataset, this will take long, but it will be done only once.".format(
                    dataset_dir
                )
            )
            self.preprocess()


        self.fnames, labels = [], []
        for txt_name in sorted(os.listdir(self.splits_dir)):
            label_name = '_'.join(txt_name.split("_")[:-2])
            f = open(os.path.join(self.splits_dir, txt_name), "r")
            for avi_name in f.readlines():
                self.fnames.append((
                    os.path.join(self.dataset_dir, "u", avi_name[:-7]),
                    os.path.join(self.dataset_dir, "v", avi_name[:-7]),
                ))
                labels.append(label_name)

        assert len(labels) == len(self.fnames)
        print("Number of {} videos: {:d}".format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {
            label: index for index, label in enumerate(sorted(set(labels)))
        }
        # Convert the list of label names into an array of label indices
        self.label_array = np.array(
            [self.label2index[label] for label in labels], dtype=int
        )

        if not os.path.exists("dataloaders/hmdb_labels.txt"):
            with open("dataloaders/hmdb_labels.txt", "w") as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id + 1) + " " + label + "\n")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        u_buffer = self.load_frames(self.fnames[index][0])
        v_buffer = self.load_frames(self.fnames[index][1])
        buffer = np.empty(u_buffer.shape[0] * 2, *u_buffer.shape[1:])
        for i in range(len(u_buffer)):
            buffer[i * 2] = u_buffer[i]
            buffer[i * 2 + 1] = v_buffer[i]
        # self.clip_len * 2 for both u and v
        buffer = self.crop(buffer, self.clip_len * 2, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == "test":
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.dataset_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, "train")):
            return False
        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Split train/val/test sets
        for file in os.listdir(self.dataset_dir + "/u"):
            if file.endswith(".bin"): continue
            file_path = os.path.join(self.dataset_dir + "/u", file)
            video_files = os.listdir(file_path)

            train_and_valid, test = train_test_split(
                video_files, test_size=0.2, random_state=42
            )
            train, val = train_test_split(
                train_and_valid, test_size=0.2, random_state=42
            )

            if self.dataset_percentage < 1.0:
                drop_percentage = 1 - self.dataset_percentage
                train, _ = train_test_split(
                    train, test_size=drop_percentage, random_state=42
                )
                val, _ = train_test_split(
                    val, test_size=drop_percentage, random_state=42
                )
                test, _ = train_test_split(
                    test, test_size=drop_percentage, random_state=42
                )

            train_dir = os.path.join(self.output_dir, "train")
            val_dir = os.path.join(self.output_dir, "val")
            test_dir = os.path.join(self.output_dir, "test")

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
                os.mkdir(os.path.join(train_dir, "u"))
                os.mkdir(os.path.join(train_dir, "v"))
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
                os.mkdir(os.path.join(val_dir, "u"))
                os.mkdir(os.path.join(val_dir, "v"))
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
                os.mkdir(os.path.join(test_dir, "u"))
                os.mkdir(os.path.join(test_dir, "v"))

            for uv in ["u", "v"]:
                for video in train:
                    dir_name = os.path.join(train_dir, uv, file)
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    copy(os.path.join(self.dataset_dir, uv, file, video), dir_name)

                for video in val:
                    dir_name = os.path.join(val_dir, uv, file)
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    copy(os.path.join(self.dataset_dir, uv, file, video), dir_name)

                for video in test:
                    dir_name = os.path.join(test_dir, uv, file)
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    copy(os.path.join(self.dataset_dir, uv, file, video), dir_name)

        print("Preprocessing finished.")

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty(
            (frame_count, self.resize_height, self.resize_width, 3), np.dtype("float32")
        )
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[
            time_index : time_index + clip_len,
            height_index : height_index + crop_size,
            width_index : width_index + crop_size,
            :,
        ]

        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # rgb_train_data = RGBDataset(
    #     dataset_dir="E:/Stanford/CS 231N/Project/ar_data/jpegs_256",
    #     splits_dir="E:/Stanford/CS 231N/Project/ar_data/hmdb51_splits",
    #     output_dir="output/", split="test", clip_len=8, preprocess=True
    # )
    # rgb_train_loader = DataLoader(rgb_train_data, batch_size=100, shuffle=True, num_workers=8)

    flow_train_data = FlowDataset(
        dataset_dir="E:/Stanford/CS 231N/Project/ar_data/tvl1_flow",
        splits_dir="E:/Stanford/CS 231N/Project/ar_data/hmdb51_splits",
        output_dir="output/", split="test", clip_len=8, preprocess=True
    )
    flow_train_loader = DataLoader(flow_train_data, batch_size=100, shuffle=True, num_workers=8)

    # for i, sample in enumerate(rgb_train_loader):
    #     inputs = sample[0]
    #     labels = sample[1]
    #     print(inputs.size())
    #     print(labels)
    #
    #     if i == 1:
    #         break
