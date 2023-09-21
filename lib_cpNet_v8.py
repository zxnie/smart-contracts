import random
import numpy as np
from lib_cpNode_v6 import RobotNode, EdgeNode
from lib_fl_v4 import DatasetScheduler
from lib_blockchain_v7 import SmartContractCpNet
from torch.utils.data import random_split


class cpNetwork:
    def __init__(self, num_edge_nodes, num_robot_nodes, dim_r_type, dim_r_num, transmission_time, fl_config,
                 encryption_switch=False, smart_switch=False, store_chain=False):
        self.rng = np.random.default_rng(69)
        self.num_edge_nodes = num_edge_nodes
        self.num_robot_nodes = num_robot_nodes
        self.dim_r_type = dim_r_type
        self.dim_r_num = dim_r_num
        self.edge_nodes = {i + 100: EdgeNode(i + 100, store_chain) for i in
                           range(num_edge_nodes)}  # Edge node id: 100, 101, ...
        self.robot_nodes = [RobotNode(i, store_chain, fl_config) for i in
                            range(num_robot_nodes)]  # Robot node id: 0, 1, ...
        '''
        Cyber network (TBD)
        '''
        # Cyber domain is fully connected wireless network
        # Currently abstracted as one node one channel, assuming one central router
        self.transmission_time = transmission_time
        '''
        Federated Learning (FL)
        '''
        self.rounds = fl_config['rounds']
        self.round_time = 5000
        self.data_scheduler = DatasetScheduler(5, self.rounds)
        self.fl_status = {'dataset_id': 0}
        self.client_datasets = None
        self.client_parameters_list = None
        self.fraction_list = None
        self.history = []
        '''
        Switches
        '''
        self.smart_switch = smart_switch
        self.encryption_switch = encryption_switch
        '''
        Stats
        '''
        self.wireless_stats = {
            't': [],
            'total_sent': [],
            'throughput_list': [],
            'latency_list': [],
        }

    def add_transaction_all(self, executor_id, transaction, env):
        for edge_id, edge_node in self.edge_nodes.items():
            # wireless transmission: executor -> other nodes
            if edge_id != executor_id:
                if executor_id < 100:
                    self.robot_nodes[executor_id].wireless_node.send_message(env, self.transmission_time)
                else:
                    self.edge_nodes[executor_id].wireless_node.send_message(env, self.transmission_time)
            edge_node.blockchain.add_new_transaction(transaction)
        for robot_id, robot_node in enumerate(self.robot_nodes):
            # wireless transmission: executor -> other nodes
            if robot_id != executor_id:
                if executor_id < 100:
                    self.robot_nodes[executor_id].wireless_node.send_message(env, self.transmission_time)
                else:
                    self.edge_nodes[executor_id].wireless_node.send_message(env, self.transmission_time)
            robot_node.blockchain.add_new_transaction(transaction)

    def add_block_all(self, executor_id, env):
        if executor_id < 100:
            new_block, proof = self.robot_nodes[executor_id].add_new_block()
            # wireless transmission: executor -> other nodes
            self.robot_nodes[executor_id].wireless_node.send_message(env, self.transmission_time)
        else:
            new_block, proof = self.edge_nodes[executor_id].add_new_block()
            # wireless transmission: executor -> other nodes
            self.edge_nodes[executor_id].wireless_node.send_message(env, self.transmission_time)
        for edge_id, edge_node in self.edge_nodes.items():
            if edge_id != executor_id:
                status = edge_node.blockchain.add_block(new_block, proof)
                # wireless transmission: other nodes -> executor
                self.edge_nodes[edge_id].wireless_node.send_message(env, self.transmission_time)
                if not status:
                    print(f'Block not added to node {edge_id}')
        for robot_id in range(self.num_robot_nodes):
            if robot_id != executor_id:
                status = self.robot_nodes[robot_id].blockchain.add_block(new_block, proof)
                # wireless transmission: other nodes -> executor
                self.robot_nodes[robot_id].wireless_node.send_message(env, self.transmission_time)
                if not status:
                    print(f'Block not added to node {robot_id}')

    def FL_net_train_step(self, round_num, executor_edge_id, executor_robot_id, env):
        print(f'Start Round {round_num + 1} ...')
        if round_num > 0:
            train_loss, valid_loss = self.FL_net_train_edge(round_num, executor_edge_id, self.client_parameters_list,
                                                            self.fraction_list, env)
        else:
            train_loss, valid_loss = None, None
        encrypted_model, key, key_hash = self.FL_net_model_share(round_num, executor_edge_id, env)
        encrypted_client_parameters_list, key_list, key_hash_list, self.fraction_list = self.FL_net_train_client(
            round_num,
            encrypted_model,
            key,
            key_hash,
            executor_robot_id,
            env
        )
        self.client_parameters_list = self.edge_nodes[executor_edge_id].decrypt_local_models(
            encrypted_client_parameters_list,
            key_list,
            self.encryption_switch,
        )

        return train_loss, valid_loss

    def FL_refresh_dataset(self, round_num):
        '''
        robot nodes refresh dataset
        actually happen in physical domain
        :param round_num:
        :return:
        '''

        flag, train_dataset_new, valid_dataset_new = self.data_scheduler.refresh_dataset(round_num)
        if flag:
            self.fl_status['dataset_id'] += 1
            self.fl_status['total_train_size'] = len(train_dataset_new)
            self.fl_status['total_dev_size'] = len(valid_dataset_new)
            samples_per_client = self.fl_status['total_train_size'] // self.num_robot_nodes
            self.client_datasets = random_split(train_dataset_new,
                                                [min(i + samples_per_client, self.fl_status['total_train_size']) - i for
                                                 i in range(0, self.fl_status['total_train_size'], samples_per_client)])
            self.fl_status['train_dataset'] = train_dataset_new
            self.fl_status['dev_dataset'] = valid_dataset_new
        return flag, train_dataset_new, valid_dataset_new

    def FL_net_train_edge(self, round_num, executor_edge_id, client_parameters_list, fraction_list, env):
        train_loss, valid_loss = self.edge_nodes[executor_edge_id].FL_edge_train_step(
            client_parameters_list,
            fraction_list,
            self.fl_status['train_dataset'],
            self.fl_status['dev_dataset'],
        )
        if round_num % 10 == 0:
            print(f'After round {round_num + 1}, train_loss = {train_loss:.4f}, dev_loss = {valid_loss:.4f}')
        return train_loss, valid_loss

    def FL_net_model_share(self, round_num, executor_edge_id, env):
        # encrypt model and add to blockchain
        encrypted_model, key, key_hash = self.edge_nodes[executor_edge_id].encrypt_model(self.encryption_switch)
        self.add_transaction_all(executor_edge_id, f'round {round_num + 1}, model from {executor_edge_id}', env)
        self.add_transaction_all(executor_edge_id, key_hash, env)
        self.add_block_all(executor_edge_id, env)
        # edge nodes synchronize model
        for i, edge_node in self.edge_nodes.items():
            if i == executor_edge_id:
                continue
            # wireless transmission: executor -> other edge nodes
            self.edge_nodes[executor_edge_id].wireless_node.send_message(env, self.transmission_time)
            edge_node.decrypt_model_and_update(encrypted_model, key, self.encryption_switch)
        return encrypted_model, key, key_hash

    def FL_net_train_client(self, round_num, encrypted_model, key, key_hash, executor_robot_id, env):
        '''
        Local train step
        '''
        flag, train_dataset_new, valid_dataset_new = self.FL_refresh_dataset(round_num)
        # robot nodes train
        encrypted_client_parameters_list = []
        key_list = []
        key_hash_list = []
        fraction_list = []
        for i, robot in enumerate(self.robot_nodes):
            encrypted_client_parameters, key_new, key_hash_new, fraction = robot.decrypt_model_and_train(
                encrypted_model,
                key,
                self.fl_status['total_train_size'],
                flag,
                self.encryption_switch,
                self.client_datasets[i],
                self.fl_status['dataset_id'],
            )
            encrypted_client_parameters_list.append(encrypted_client_parameters)
            self.add_transaction_all(executor_robot_id,
                                     f'round {round_num}, client id: {i}, dataset id: {robot.fl_client.dataset_id}, dataset volume: {fraction}, key hash: {key_hash_new}',
                                     env)
            key_list.append(key_new)
            key_hash_list.append(key_hash_new)
            fraction_list.append(fraction)
        self.add_block_all(executor_robot_id, env)  # add local training model hashes to blockchain
        return encrypted_client_parameters_list, key_list, key_hash_list, fraction_list

    def FL(self, env):
        sc_cpnet = SmartContractCpNet(self.num_edge_nodes, self.num_robot_nodes)
        executor_edge = sc_cpnet.executor_edge_id
        executor_robot = sc_cpnet.executor_robot_id
        round_num = 0
        while True:
            train_loss, valid_loss = self.FL_net_train_step(round_num, executor_edge, executor_robot, env)
            # FL performance
            self.history.append((train_loss, valid_loss))
            # Ready for next iteration
            executor_edge = sc_cpnet.delegate_edge()
            executor_robot = sc_cpnet.delegate_robot()
            round_num += 1
            yield env.timeout(self.round_time)
            # if round_num >= self.rounds:
            #     break

    def record_wireless_stats(self, env):
        while True:
            yield env.timeout(10)
            # wireless performance
            total_sent_edge = sum(node.wireless_node.sent_messages for node in self.edge_nodes.values())
            total_sent_robots = sum(node.wireless_node.sent_messages for node in self.robot_nodes)
            total_sent = total_sent_edge + total_sent_robots
            throughput = total_sent
            # Compute average latency
            latencies = []
            for node in self.edge_nodes.values():
                latencies += node.wireless_node.latencies
            for node in self.robot_nodes:
                latencies += node.wireless_node.latencies
            # avg_latency = sum(latencies) / total_sent if total_sent > 0 else 0
            # print('total sent', total_sent)
            # print('number of latencies', len(latencies))
            self.wireless_stats['t'].append(env.now)
            self.wireless_stats['total_sent'].append(total_sent)
            self.wireless_stats['throughput_list'].append(throughput)
            # self.wireless_stats['latency_list'].append(avg_latency)
            self.wireless_stats['latency_list'] += latencies
            for node in self.edge_nodes.values():
                node.wireless_node.reset_stats()
            for node in self.robot_nodes:
                node.wireless_node.reset_stats()
