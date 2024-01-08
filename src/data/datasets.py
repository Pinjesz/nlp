import pickle
from torch.utils import data


class TrainDataset(data.Dataset):
    def __init__(self, tokenized_data_path) -> None:
        super().__init__()
        # data = get_tokenized_data("bert-base-cased")
        # tokenized_data_path = "data/tokenized/train.pkl"

        with open(tokenized_data_path, "rb") as file:
            data = pickle.load(file)

        self.x = data["tokens"]
        self.y = data["labels"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class TrialDataset(data.Dataset):
    def __init__(self, tokenized_data_path) -> None:
        super().__init__()

        with open(tokenized_data_path, "rb") as file:
            data = pickle.load(file)

        self.x = data["tokens"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]
