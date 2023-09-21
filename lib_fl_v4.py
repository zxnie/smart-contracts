import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# %%
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class WM38SplitDataset(Dataset):
    def __init__(self, A_list, D_list):
        self.A_list = A_list.astype(np.float32)
        self.D_list = D_list.astype(np.float32)

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        A = self.A_list[idx]
        D = self.D_list[idx]
        return A, D


class DeviceDataLoader:
    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, device)

    def __len__(self):
        return len(self.dl)


def Euclidean_distance(C, D):
    return torch.norm(C - D, p=2)


class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, C, D):
        assert C.shape == D.shape, 'Input tensors must have the same shape'
        # Initialize variables
        num_matrices = C.shape[0]
        total_distance = 0
        # Iterate through all matrices in C and D
        for i in range(num_matrices):
            for j in range(num_matrices):
                # Calculate Euclidean distance between matrices i and j
                dist = Euclidean_distance(C[i], D[j])
                # Add distance to total
                total_distance += dist
        # Calculate average distance
        avg_distance = total_distance / (num_matrices ** 2)
        # avg_distance = torch.sqrt(torch.sum((C - D) ** 2)).mean()
        return avg_distance


class FederatedNet(torch.nn.Module):
    def __init__(self):
        super(FederatedNet, self).__init__()
        self.fc1 = nn.Linear(52 * 52, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 52 * 52)
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2, 'fc3': self.fc3}

    def forward(self, x):
        out = x.view(-1, 52 * 52)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out.view(-1, 52, 52)

    def get_track_layers(self):
        return self.track_layers

    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data,
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def process_batch(self, batch):
        loss_function = EuclideanDistanceLoss()
        B = self(batch[0])
        C = batch[0] + B
        loss = loss_function(C, batch[1])
        return loss

    def fit(self, dataset, num_epochs, lr, batch_size=32, opt=torch.optim.SGD):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size, shuffle=True))
        optimizer = opt(self.parameters(), lr)
        history_loss = []
        for _ in range(num_epochs):
            losses = []
            for batch in dataloader:
                loss = self.process_batch(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss.detach()
                losses.append(loss)
            avg_loss = torch.stack(losses).mean().item()
            history_loss.append(avg_loss)
        return history_loss

    def evaluate(self, dataset, batch_size=32):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size))
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                loss = self.process_batch(batch)
                losses.append(loss)
        avg_loss = torch.stack(losses).mean().item()
        return avg_loss


class FLClient:
    def __init__(self, client_id, epochs_per_client, learning_rate, batch_size):
        self.client_id = client_id
        self.dataset = None
        self.dataset_id = None
        self.epochs_per_client = epochs_per_client
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def refresh_dataset(self, dataset, dataset_id):
        self.dataset = dataset
        self.dataset_id = dataset_id

    def get_dataset_size(self):
        return len(self.dataset)

    def get_client_id(self):
        return self.client_id

    def train(self, parameters_dict):
        net = to_device(FederatedNet(), device)
        net.apply_parameters(parameters_dict)
        history_loss = net.fit(self.dataset, self.epochs_per_client, self.learning_rate, self.batch_size)
        # print(f'{self.client_id}: Loss = {history_loss[-1]:.4f}')
        return net.get_parameters()


class DatasetScheduler:
    def __init__(self, interval, max_rounds):
        self.interval = interval
        self.current_dataset_number = 0
        self.dataset_change_rounds = np.arange(0, max_rounds, interval)

    def load_dataset(self):
        if self.current_dataset_number == 38:
            self.current_dataset_number = 1
        elif self.current_dataset_number == 3:
            self.current_dataset_number = 4
        data = np.load(f'dataset/split_{self.current_dataset_number}.npz')
        # Assuming your dataset A and desired outcome D are numpy arrays
        A = data['data']
        D = data['template']

        train_dataset = WM38SplitDataset(A, D)
        train_dataset_size = int(len(train_dataset) * 0.83)
        dev_dataset_size = len(train_dataset) - train_dataset_size
        train_dataset, dev_dataset = random_split(train_dataset, [train_dataset_size, dev_dataset_size])
        return train_dataset, dev_dataset

    def refresh_dataset(self, round_number):
        if round_number in self.dataset_change_rounds:
            print('Dataset changed')
            self.current_dataset_number += 1
            train_dataset, dev_dataset = self.load_dataset()
            return True, train_dataset, dev_dataset
        return False, None, None


class FedNetSystem:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.rounds = 50
        self.batch_size = 32
        self.epochs_per_client = 5
        self.learning_rate = 1e-2
        self.clients = [FLClient(
            f'client_{i}',
            self.epochs_per_client,
            self.learning_rate,
            self.batch_size,
        ) for i in range(num_clients)]
        self.global_net = to_device(FederatedNet(), device)
        self.data_scheduler = DatasetScheduler(10, self.rounds)

    def train(self):
        history = []
        for i in range(self.rounds):
            # print(f'Start Round {i + 1} ...')
            # global model
            curr_parameters = self.global_net.get_parameters()
            # initialize local model dict
            new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
            # refresh dataset
            flag, train_dataset1, dev_dataset1 = self.data_scheduler.refresh_dataset(i)
            if flag:
                total_train_size = len(train_dataset1)
                total_dev_size = len(dev_dataset1)
                examples_per_client = total_train_size // self.num_clients
                client_datasets = random_split(train_dataset1,
                                               [min(i + examples_per_client, total_train_size) - i for i in
                                                range(0, total_train_size, examples_per_client)])
                for i, client in enumerate(self.clients):
                    client.refresh_dataset(client_datasets[i], i)
                train_dataset = train_dataset1
                dev_dataset = dev_dataset1
            # train local model
            for client in self.clients:
                # Align local model with global model and perform a training step
                client_parameters = client.train(curr_parameters)
                # Aggregate local model with train dataset size as weight
                fraction = client.get_dataset_size() / total_train_size
                for layer_name in client_parameters:
                    new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
                    new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
            # update global model
            self.global_net.apply_parameters(new_parameters)
            # evaluate global model
            train_loss = self.global_net.evaluate(train_dataset)
            dev_loss = self.global_net.evaluate(dev_dataset)
            # print(f'After round {i + 1}, train_loss = {train_loss:.4f}, dev_loss = {dev_loss:.4f}\n')
            history.append((train_loss, dev_loss))
        return history


device = get_device()
# %%
if __name__ == '__main__':
    FL_instance = FedNetSystem(10)
    history = FL_instance.train()
