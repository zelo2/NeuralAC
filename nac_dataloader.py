# author@Zelo2
from torch.utils.data import Dataset, DataLoader



class nac_dataset(Dataset):
    def __init__(self, data):
        super(nac_dataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return [self.data[index, :-1], self.data[index, -1]]  # train data, label
