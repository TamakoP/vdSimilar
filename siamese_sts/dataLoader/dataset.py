from torch.utils.data import Dataset

"""
Standard Pytorch Dataset class for loading datasets.
"""


class SiameseDataset(Dataset):
    def __init__(
        self, dataset
    ):
        """
        initializes  and populates  the length, data and target tensors, and raw texts list
        """

        self.sent1_tensor = []
        self.sent2_tensor = []
        self.target_tensor = []
        self.sents1_length_tensor = []
        self.sents2_length_tensor = []
        for i in range(len(dataset)):
            self.sent1_tensor.append(dataset[i][0])
            self.sent2_tensor.append(dataset[i][1])
            self.target_tensor.append(dataset[i][2])
            self.sents1_length_tensor.append(dataset[i][3])
            self.sents2_length_tensor.append(dataset[i][4])



    def __getitem__(self, index):
        """
        returns the tuple of data tensor, targets, lengths of sequences tensor and raw texts list
        """
        return (
            self.sent1_tensor[index],
            self.sent2_tensor[index],
            self.sents1_length_tensor[index],
            self.sents2_length_tensor[index],
            self.target_tensor[index],

        )

    def __len__(self):
        """
        returns the length of the data tensor.
        """
        return len(self.target_tensor)
