from copy import deepcopy
import numpy as np
import torch
from lib_wireless_v2 import WirelessNode
from lib_blockchain_v7 import Blockchain
from lib_blockchain_v7 import SmartContractEdge, SmartContractRobot
from lib_fl_v4 import FLClient, device, to_device, FederatedNet


class Node:
    def __init__(self, id, store_chain):
        self.id = id
        self.rng = np.random.default_rng(seed=id + 100)
        self.wireless_node = WirelessNode(id, seed=200 + id)
        self.blockchain = Blockchain(id, {}, store_chain)

    def add_new_block(self):
        new_block_ind, proof = self.blockchain.mine()
        new_block = deepcopy(self.blockchain.chain[new_block_ind])
        new_block.hash = None
        return new_block, proof


class RobotNode(Node):
    def __init__(self, id, store_chain, fl_config):
        super().__init__(id, store_chain)
        # Cyber domain
        self.fl_client = FLClient(
            f'client_{id}',
            fl_config['epochs_per_client'],
            fl_config['learning_rate'],
            fl_config['batch_size'],
        )
        self.smart_contract = SmartContractRobot(id)

        self.key_part = None

    def FL_robot_train_step(self, parameters_dict, total_train_size, flag_refresh_dataset, dataset=None, dataset_id=None):
        if flag_refresh_dataset:
            self.fl_client.refresh_dataset(dataset, dataset_id)
        client_parameters = self.fl_client.train(parameters_dict)
        fraction = self.fl_client.get_dataset_size() / total_train_size
        return client_parameters, fraction

    def decrpy_model(self, encrypted_model, key, encryption_switch):
        key_hash = self.blockchain.chain[-2]['transactions'][-1]
        return self.smart_contract.decrypt(encrypted_model, key, key_hash, encryption_switch)

    def decrypt_model_and_train(self, encrypted_model, key, total_train_size, flag_refresh_dataset,
                                encryption_switch, dataset=None, dataset_id=None):
        key_hash = self.blockchain.chain[-1].transactions[-1]
        curr_parameters = self.smart_contract.decrypt(encrypted_model, key, key_hash, encryption_switch)
        if encryption_switch:
            for layer_name in curr_parameters:
                curr_parameters[layer_name]['weight'] = torch.tensor(curr_parameters[layer_name]['weight'])
                curr_parameters[layer_name]['bias'] = torch.tensor(curr_parameters[layer_name]['bias'])
        client_parameters, fraction = self.FL_robot_train_step(curr_parameters, total_train_size, flag_refresh_dataset, dataset,
                                                               dataset_id)
        encrypted_client_parameters, key, key_hash = self.encrypt_model(client_parameters, encryption_switch)
        return encrypted_client_parameters, key, key_hash, fraction

    def encrypt_model(self, client_parameters, encryption_switch):
        return self.smart_contract.encrypt(client_parameters, encryption_switch)


class EdgeNode(Node):
    def __init__(self, id, store_chain):
        super().__init__(id, store_chain)
        # Cyber domain
        self.global_net = to_device(FederatedNet(), device)
        self.smart_contract = SmartContractEdge(id)
        self.key_part = None

    def FL_edge_train_step(self, client_parameters_list, fraction_list, train_dataset, dev_dataset):
        # global model
        curr_parameters = self.global_net.get_parameters()
        # initialize local model dict
        new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
        # train local model
        for (client_parameters, fraction) in zip(client_parameters_list, fraction_list):
            for layer_name in client_parameters:
                new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
                new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
        # update global model
        self.global_net.apply_parameters(new_parameters)
        # evaluate global model
        train_loss = self.global_net.evaluate(train_dataset)
        dev_loss = self.global_net.evaluate(dev_dataset)

        return train_loss, dev_loss

    def encrypt_model(self, encryption_switch):
        return self.smart_contract.encrypt(self.global_net.get_parameters(), encryption_switch)

    def decrypt_model_and_update(self, encrypted_model, key, encryption_switch):
        key_hash = self.blockchain.chain[-1].transactions[-1]
        curr_parameters = self.smart_contract.decrypt(encrypted_model, key, key_hash, encryption_switch)
        if encryption_switch:
            for layer_name in curr_parameters:
                curr_parameters[layer_name]['weight'] = torch.tensor(curr_parameters[layer_name]['weight'])
                curr_parameters[layer_name]['bias'] = torch.tensor(curr_parameters[layer_name]['bias'])
        self.global_net.apply_parameters(curr_parameters)

    def decrypt_local_models(self, encrypted_client_parameters_list, key_list, encryption_switch):
        client_parameters_list = []
        for i, encrypted_client_parameters in enumerate(encrypted_client_parameters_list):
            key = key_list[i]
            key_hash = self.blockchain.chain[-1].transactions[i][-64:]  # A SHA-256 hash is 256 bits, 64 hexadecimal digits.
            client_parameters = self.smart_contract.decrypt(encrypted_client_parameters, key, key_hash, encryption_switch)
            if encryption_switch:
                for layer_name in client_parameters:
                    client_parameters[layer_name]['weight'] = torch.tensor(client_parameters_list[-1][layer_name]['weight'])
                    client_parameters[layer_name]['bias'] = torch.tensor(client_parameters_list[-1][layer_name]['bias'])
            client_parameters_list.append(client_parameters)
        return client_parameters_list
