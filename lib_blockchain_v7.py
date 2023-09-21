from copy import deepcopy
from hashlib import sha256
import json
import time
import numpy as np
from datetime import datetime
from lib_secret_sharing_v8 import generate_key, encrypt_dict, calculate_hash_secret, decrypt_dict


class Block:
    def __init__(self, index, transactions, timestamp, previous_hash):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.hash = None
        self.previous_hash = previous_hash
        self.nonce = 0


class Blockchain:
    def __init__(self, id, initial_message, store_chain=True):
        self.unconfirmed_transactions = []
        time_str = datetime.now().strftime('%Y%m%d%H%M%S')
        self.chain_fn = f'blockchain_{id}_{time_str}.json'
        self.chain = []
        self.store_chain = store_chain
        # difficulty of our PoW algorithm
        self.difficulty = 1
        self.create_genesis_block(initial_message)

    def compute_hash(self, block):
        '''
        A function that return the hash of the block contents.
        '''
        block_string = json.dumps(block.__dict__, sort_keys=True, cls=BlockEncoder)
        return sha256(block_string.encode()).hexdigest()

    def create_genesis_block(self, trust_dict):
        '''
        A function to generate genesis block and appends it to
        the chain. The block has index 0, previous_hash as 0, and
        a valid hash.
        '''
        genesis_block = Block(0, [trust_dict], None, '0')
        genesis_block.hash = self.compute_hash(genesis_block)
        self.chain.append(genesis_block)
        if self.store_chain:
            append_to_json_file(self.chain_fn, genesis_block)

    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, block, proof):
        '''
        A function that adds the block to the chain after verification.
        Verification includes:
        * Checking if the proof is valid.
        * The previous_hash referred in the block and the hash of latest block
          in the chain match.
        '''
        previous_hash = self.last_block.hash

        # Previous hash
        if previous_hash != block.previous_hash:
            return False

        # Difficulty check
        if not proof.startswith('0' * self.difficulty):
            return False

        # Proof check
        if not proof == self.compute_hash(block):
            return False

        new_block = deepcopy(block)
        new_block.hash = proof
        self.chain.append(new_block)
        if self.store_chain:
            append_to_json_file(self.chain_fn, new_block)

        self.unconfirmed_transactions = []

        return True

    def proof_of_work(self, block):
        '''
        Function that tries different values of nonce to get a hash
        that satisfies our difficulty criteria.
        '''
        block.nonce = 0

        computed_hash = self.compute_hash(block)
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = self.compute_hash(block)

        return computed_hash

    def add_new_transaction(self, transaction):
        self.unconfirmed_transactions.append(transaction)

    def mine(self):
        '''
        This function serves as an interface to add the pending
        transactions to the blockchain by adding them to the block
        and figuring out Proof Of Work.
        '''
        if not self.unconfirmed_transactions:
            return False

        last_block = self.last_block

        new_block = Block(index=last_block.index + 1,
                          transactions=self.unconfirmed_transactions,
                          timestamp=time.time(),
                          previous_hash=last_block.hash)

        proof = self.proof_of_work(new_block)
        self.add_block(new_block, proof)

        return new_block.index, proof

    def save_as_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chain, f, cls=BlockEncoder)

    def get_latest_trust(self):
        return self.chain[-1].transactions[0]


class BlockEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Block):
            return obj.__dict__
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super().default(obj)


def append_to_json_file(filename, item):
    with open(filename, 'a') as f:
        f.write(json.dumps(item, cls=BlockEncoder))
        f.write('\n')


def load_from_json_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)


class SmartContractCpNet:
    '''
    Delegate edge node to distribute model parameters
    TBD: Delegate robot node to distribute keys
    '''

    def __init__(self, num_edge_nodes, num_robot_nodes):
        self.num_edge_nodes = num_edge_nodes
        self.num_robot_nodes = num_robot_nodes
        self.executor_edge_id = 100
        self.executor_robot_id = 0

    def delegate_edge(self):
        if self.executor_edge_id == 100 + self.num_edge_nodes - 1:
            self.executor_edge_id = 100
        else:
            self.executor_edge_id += 1
        return self.executor_edge_id

    def delegate_robot(self):
        if self.executor_robot_id == self.num_robot_nodes - 1:
            self.executor_robot_id = 0
        else:
            self.executor_robot_id += 1
        return self.executor_robot_id


class SmartContractNode:
    '''
    Encrypt model parameters and add to blockchain
    Distribute keys to robot nodes
    '''

    def __init__(self, id):
        self.node_id = id

    def encrypt(self, curr_parameters, encryption_switch):
        key = generate_key()
        # print(f'Node {self.node_id} encrypts model with key {key}')
        if encryption_switch:
            encrypted = encrypt_dict(curr_parameters, key)
        else:
            encrypted = curr_parameters
        key_hash = calculate_hash_secret(key)
        return encrypted, key, key_hash

    def decrypt(self, encrypted_model, key, key_hash, encryption_switch):
        if calculate_hash_secret(key) == key_hash:
            if encryption_switch:
                decrypted = decrypt_dict(encrypted_model, key)
            else:
                decrypted = encrypted_model
        else:
            decrypted = None
            raise ValueError('Key hash does not match')
        return decrypted

    def calculate_hash_secret(self, secret):
        return calculate_hash_secret(secret)


class SmartContractEdge(SmartContractNode):
    def __init__(self, id):
        super().__init__(id)


class SmartContractRobot(SmartContractNode):
    def __init__(self, id):
        super().__init__(id)


if __name__ == '__main__':

    blockchain = Blockchain(0, {})


    def get_chain():
        chain_data = []
        for block in blockchain.chain:
            chain_data.append(block.__dict__)
        return json.dumps({'length': len(chain_data),
                           'chain': chain_data})
