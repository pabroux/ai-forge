import torch


class AudioDataSet(torch.utils.data.Dataset):
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __init__(
        self, path_ground_truth, path_input, transform=None, target_transform=None
    ):
        # self.ground_truth=pd.read_csv(path_ground_truth)
        # self.input = path_input
        # self.transform = transform
        # self.target_transform = target_transform
        pass

    def __len__(self):
        # return len(self.img_labels)
        pass

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #    image = self.transform(image)
        # if self.target_transform:
        #    label = self.target_transform(label)
        # return image, label
        pass
