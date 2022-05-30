from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, encodings):
        super().__init__()

        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}

        return item
