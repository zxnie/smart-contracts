# %%
import json
import pickle
import numpy as np
import simpy
from lib_cpNet_v8 import cpNetwork


class Simulator:
    def __init__(self, config, transmission_time=2, smart_switch=False):
        # Simulation parameters
        self.num_edge_nodes = config['num_edge_nodes']
        self.num_robot_nodes = config['num_robot_nodes']
        self.simulation_time = config['simulation_time']
        self.simulation_config = config
        self.transmission_time = transmission_time
        self.transmission_interval = 500
        self.report_interval = 1000
        self.encryption_switch = False
        self.smart_switch = smart_switch
        self.store_chain = False
        self.fl_config = {
            'rounds': 200,
            'epochs_per_client': 5,
            'learning_rate': 1e-3,
            'batch_size': 32
        }

        # Create environment, channels and nodes
        self.env = simpy.Environment(0)
        self.cpNet = cpNetwork(
            self.num_edge_nodes,
            self.num_robot_nodes,
            self.simulation_config['dim_r_type'],
            self.simulation_config['dim_r_num'],
            self.transmission_time,
            self.fl_config,
            self.encryption_switch,
            self.smart_switch,
            self.store_chain
        )

    def run(self):
        # self.cpNet.FL(self.env)

        self.env.process(self.cpNet.FL(self.env))
        self.env.process(self.cpNet.record_wireless_stats(self.env))
        self.env.run(until=self.simulation_time)

        return self.cpNet.history, self.cpNet.wireless_stats


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# %%
if __name__ == '__main__':
    simulation_time = 100000
    # num_edge_node_list = [4, 4, 4, 5, 5, 6, 6]
    # num_robot_node_list = [9, 13, 19, 23, 29, 31, 37]
    # average_delay_list = [5.59855709, 5.64527724, 5.73363911, 5.82211672, 5.99330065, 6.09793224, 6.41913226]
    num_edge_node_list = [4]
    num_robot_node_list = [9]
    average_delay_list = [5.59855709]
    for num_edge_nodes, num_robot_nodes, transmission_time in zip(num_edge_node_list, num_robot_node_list, average_delay_list):
        # num_edge_nodes = 6
        # num_robot_nodes = 37
        config = {
            'num_edge_nodes': num_edge_nodes,
            'num_robot_nodes': num_robot_nodes,
            'simulation_time': simulation_time,
            'dim_r_type': 3,
            'dim_r_num': 3,
        }
        my_simlator = Simulator(config, transmission_time, False)
        history, wireless_stats = my_simlator.run()

        results = {
            'fl_history': {
                'x': my_simlator.cpNet.data_scheduler.dataset_change_rounds,
                'train_loss': [history[i][0] for i in range(len(history))],
                'valid_loss': [history[i][1] for i in range(len(history))],
            },
            'wireless_stats': wireless_stats
        }

        fn = f'{num_edge_nodes}edge_{num_robot_nodes}robots_{simulation_time}ms_latency_new'
        with open(f'results_{fn}.pkl', 'wb') as f:
            pickle.dump(results, f)

        my_simlator.cpNet.robot_nodes[0].blockchain.save_as_json(f'blockchain_{fn}.json')

        print(f'Simulation of {num_edge_nodes}edge_{num_robot_nodes}robots complete, results saved.')
