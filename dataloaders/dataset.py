import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from shutil import copytree
import torch.nn.functional as F
from args import get_args

args = get_args()

RESIZE_HEIGHT = 64
RESIZE_WIDTH = 85
CROP_SIZE = 60

def temporal_padding(buffer, clip_len):
    """Pad buffer to have temporal length of clip_len, Pad with 0"""
    if buffer.shape[0] > clip_len:
        pass
    else:
        pad_len = clip_len - buffer.shape[0] + 1
        npad = ((pad_len, 0), (0, 0), (0, 0), (0,0))
        # buffer = np.pad(buffer, pad_width=npad, mode='constant', constant_values=0)
        buffer = np.pad(buffer, pad_width=npad, mode='mean')
    assert buffer.shape[0] - clip_len > 0, "Incorrect Padding"
    return buffer

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
        self.dataset_dir, self.splits_dir, self.output_dir = (
            dataset_dir,
            splits_dir,
            output_dir,
        )
        self.dataset_percentage = dataset_percentage
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = RESIZE_HEIGHT
        self.resize_width = RESIZE_WIDTH
        self.crop_size = CROP_SIZE

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
        data_dir = os.path.join(self.output_dir, self.split)
        unique_labels = os.listdir(data_dir)
        for label in unique_labels:
            label_dir = os.path.join(data_dir, label)
            video_names = os.listdir(label_dir)
            for video_name in video_names:
                self.fnames.append(os.path.join(label_dir, video_name))
                labels.append(label)

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

        if args.skip_frames:
            print("===== Skipping Frames =====") 

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        # Skip frames
        if args.skip_frames:
            buffer = buffer[0::2, :, :]            
        labels = np.array(self.label_array[index])

        if self.split == "test" or "train":
            # print("Augmentation Performed in ", self.split)
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

        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")
        test_dir = os.path.join(self.output_dir, "test")

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        # Split train/val/test sets
        label_video_matrix = []
        label_index_name = []
        for txt_name in sorted(os.listdir(self.splits_dir)):
            label_name = "_".join(txt_name.split("_")[:-2])
            if not os.path.exists(os.path.join(train_dir, label_name)):
                os.mkdir(os.path.join(train_dir, label_name))
                os.mkdir(os.path.join(val_dir, label_name))
                os.mkdir(os.path.join(test_dir, label_name))
                label_video_matrix.append([])
                label_index_name.append(label_name)

            f = open(os.path.join(self.splits_dir, txt_name), "r")
            for avi_name in f.readlines():
                video_name = avi_name.split(".")[0]
                label_video_matrix[-1].append(video_name)

        for label_index in range(len(label_video_matrix)):
            train_and_valid, test = train_test_split(
                list(set(label_video_matrix[label_index])),
                test_size=0.2,
                random_state=42,
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

            label_name = label_index_name[label_index]
            for video in train:
                dir_name = os.path.join(train_dir, label_name, video)
                copytree(os.path.join(self.dataset_dir, video), dir_name)
                self.process_frames_under_video_dir(dir_name)

            for video in val:
                dir_name = os.path.join(val_dir, label_name, video)
                copytree(os.path.join(self.dataset_dir, video), dir_name)
                self.process_frames_under_video_dir(dir_name)

            for video in test:
                dir_name = os.path.join(test_dir, label_name, video)
                copytree(os.path.join(self.dataset_dir, video), dir_name)
                self.process_frames_under_video_dir(dir_name)

        print("Preprocessing finished.")

    def process_frames_under_video_dir(self, video_dir):
        for img_name in os.listdir(video_dir):
            img_path = os.path.join(video_dir, img_name)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(
                img,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_AREA,
            )
            assert cv2.imwrite(img_path, resized_img)

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
        buffer = temporal_padding(buffer, clip_len)
        assert buffer.shape[0] - clip_len > 0, "Incorrect Padding"
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
            in_channel: The frame window size for both u and v (i.e. we adopt in_channel frames from u and v
            respectively so there are totally 2 * in_channel channels for the input to C3D).
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(
        self,
        dataset_dir,
        splits_dir,
        output_dir,
        in_channel=10,
        dataset_percentage=1.0,
        split="train",
        clip_len=16,
        preprocess=False,
    ):
        self.dataset_dir, self.splits_dir, self.output_dir = (
            dataset_dir,
            splits_dir,
            output_dir,
        )
        self.in_channel = in_channel
        self.dataset_percentage = dataset_percentage
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = RESIZE_HEIGHT
        self.resize_width = RESIZE_WIDTH
        self.crop_size = CROP_SIZE

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
        data_dir = os.path.join(self.output_dir, self.split)
        unique_labels = os.listdir(os.path.join(data_dir, "u"))
        for label in unique_labels:
            label_dir = os.path.join(data_dir, "u", label)
            video_names = os.listdir(label_dir)
            for video_name in video_names:
                self.fnames.append(
                    (
                        os.path.join(data_dir, "u", label, video_name),
                        os.path.join(data_dir, "v", label, video_name),
                    )
                )
                labels.append(label)

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
        min_frame_size = min(
            [
                len(os.listdir(self.fnames[min(index + i, len(self.fnames) - 1)][0]))
                for i in range(self.in_channel)
            ]
        )
        buffer = np.empty(
            (
                self.in_channel * 2,
                min_frame_size,
                self.resize_height,
                self.resize_width,
            ),
            np.dtype("float32"),
        )
        for i in range(self.in_channel):
            # buffer format before squeeze: [num_frame x H x W x C]
            # Since each pixel of grayscale images have the same value across R, G, B channels, only keep one of them.
            u_buffer = np.squeeze(
                self.load_frames(self.fnames[min(index + i, len(self.fnames) - 1)][0])[
                    :, :, :, 0
                ]
            )
            v_buffer = np.squeeze(
                self.load_frames(self.fnames[min(index + i, len(self.fnames) - 1)][1])[
                    :, :, :, 0
                ]
            )
            buffer[i * 2] = u_buffer[:min_frame_size, :, :]
            buffer[i * 2 + 1] = v_buffer[:min_frame_size, :, :]

        # [C x num_frame x H x W] --> [num_frame x H x W x C], where C is frame window size.
        buffer = np.array(buffer).transpose((1, 2, 3, 0))

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == "train":
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

        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")
        test_dir = os.path.join(self.output_dir, "test")

        dirs = []
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            dirs.append(os.path.join(train_dir, "u"))
            os.mkdir(dirs[-1])
            dirs.append(os.path.join(train_dir, "v"))
            os.mkdir(dirs[-1])
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
            dirs.append(os.path.join(val_dir, "u"))
            os.mkdir(dirs[-1])
            dirs.append(os.path.join(val_dir, "v"))
            os.mkdir(dirs[-1])
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            dirs.append(os.path.join(test_dir, "u"))
            os.mkdir(dirs[-1])
            dirs.append(os.path.join(test_dir, "v"))
            os.mkdir(dirs[-1])

        # Split train/val/test sets
        label_video_matrix = []
        label_index_name = []
        for txt_name in sorted(os.listdir(self.splits_dir)):
            label_name = "_".join(txt_name.split("_")[:-2])
            if not os.path.exists(os.path.join(dirs[0], label_name)):
                for dir in dirs:
                    os.mkdir(os.path.join(dir, label_name))
                label_video_matrix.append([])
                label_index_name.append(label_name)

            f = open(os.path.join(self.splits_dir, txt_name), "r")
            for avi_name in f.readlines():
                video_name = avi_name.split(".")[0]
                label_video_matrix[-1].append(video_name)

        for label_index in range(len(label_video_matrix)):
            train_and_valid, test = train_test_split(
                list(set(label_video_matrix[label_index])),
                test_size=0.2,
                random_state=42,
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

            label_name = label_index_name[label_index]
            for uv in ["u", "v"]:
                for video in train:
                    dir_name = os.path.join(train_dir, uv, label_name, video)
                    copytree(os.path.join(self.dataset_dir, uv, video), dir_name)
                    self.process_frames_under_video_dir(dir_name)

                for video in val:
                    dir_name = os.path.join(val_dir, uv, label_name, video)
                    copytree(os.path.join(self.dataset_dir, uv, video), dir_name)
                    self.process_frames_under_video_dir(dir_name)

                for video in test:
                    dir_name = os.path.join(test_dir, uv, label_name, video)
                    copytree(os.path.join(self.dataset_dir, uv, video), dir_name)
                    self.process_frames_under_video_dir(dir_name)

        print("Preprocessing finished.")

    def process_frames_under_video_dir(self, video_dir):
        for img_name in os.listdir(video_dir):
            img_path = os.path.join(video_dir, img_name)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(
                img,
                (self.resize_width, self.resize_height),
                interpolation=cv2.INTER_AREA,
            )
            assert cv2.imwrite(img_path, resized_img)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
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
        buffer = temporal_padding(buffer, clip_len)
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
        dataset_dir="./data/tvl1_flow",
        splits_dir="./fixtures/hmdb51_splits",
        output_dir="output/",
        split="test",
        clip_len=8,
        preprocess=True,
    )
    flow_train_loader = DataLoader(
        flow_train_data, batch_size=100, shuffle=True, num_workers=8
    )

    # for i, sample in enumerate(rgb_train_loader):
    #     inputs = sample[0]
    #     labels = sample[1]
    #     print(inputs.size())
    #     print(labels)
    #
    #     if i == 1:
    #         break
