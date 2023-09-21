import hashlib
import networkx as nx
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import json
import math
import torch


def calculate_hash_secret(data):
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class MerkleTree:
    def __init__(self, leaf_node_data_dict):
        self.G = nx.DiGraph()
        self.G_stripped = nx.DiGraph()
        self.sibling_dict = None
        self.root = self.generate_merkle_tree(leaf_node_data_dict)

    def merkle_tree(self, leaf_node_ids, depth=0):
        if len(leaf_node_ids) == 1:
            return leaf_node_ids[0]

        parents_id = []
        node_number = 1
        for i in range(0, len(leaf_node_ids), 2):
            node1_id = leaf_node_ids[i]
            if i + 1 < len(leaf_node_ids):
                node2_id = leaf_node_ids[i + 1]
            else:
                node2_id = node1_id
            parent_id = f'{depth + 1}_{node_number}'
            parents_id.append(parent_id)
            parent_data = self.G.nodes[node1_id]['hash'] + self.G.nodes[node2_id]['hash']
            self.G.add_nodes_from([(parent_id, {'hash': calculate_hash_secret(parent_data)})])
            self.G.add_edge(parent_id, node1_id)
            self.G.add_edge(parent_id, node2_id)
            self.G_stripped.add_nodes_from([parent_id])
            self.G_stripped.add_edge(parent_id, node1_id)
            self.G_stripped.add_edge(parent_id, node2_id)
            node_number += 1
        return self.merkle_tree(parents_id, depth + 1)

    def generate_merkle_tree(self, leaf_node_data_dict):
        self.sibling_dict = self.get_sibling_dict(len(leaf_node_data_dict))
        for node_id, data in leaf_node_data_dict.items():
            self.G.add_nodes_from([(node_id, {'hash': calculate_hash_secret(data)})])
            self.G_stripped.add_nodes_from([node_id])
        root_id = self.merkle_tree(list(leaf_node_data_dict.keys()))
        return root_id

    def get_hashes_to_root(self, node_id):
        path = nx.shortest_path(self.G, source=self.root, target=node_id)
        siblings = {}
        self._get_hashes_to_root_recursive(self.root, node_id, path[1:], siblings)
        return siblings, path

    def _get_hashes_to_root_recursive(self, node_id, user_id, path, siblings):
        if node_id is None:
            return None

        child_nodes = sorted(list(self.G.successors(node_id)))
        if len(child_nodes) == 2:
            # Have both left and right
            child_left = child_nodes[0]
            child_right = child_nodes[1]
        else:
            # Have only left
            child_left = child_nodes[0]
            child_right = child_nodes[0]

        if child_left in path:
            if child_right not in path:
                siblings[child_right] = self.G.nodes[child_right]['hash']
            if child_left == user_id:
                return True
            else:
                return self._get_hashes_to_root_recursive(child_left, user_id, path, siblings)

        if child_right in path:
            if child_left not in path:
                siblings[child_left] = self.G.nodes[child_left]['hash']
            if child_right == user_id:
                return True
            else:
                return self._get_hashes_to_root_recursive(child_right, user_id, path, siblings)

    def get_sibling_dict(self, N):
        sibling_dict = {}
        for i in range(0, N, 2):
            # Handle the case where there is an odd number of nodes
            if i + 1 >= N:
                sibling_dict[i] = i
            else:
                sibling_dict[i] = i + 1
                sibling_dict[i + 1] = i
        return sibling_dict


class SecretRecipient:
    def __init__(self, id, root_id, merkle_tree_stripped, hashes):
        self.id = id
        self.root_id = root_id
        self.merkle_tree_stripped = merkle_tree_stripped  # networkx digraph object
        self.hashes = hashes

    def calculate_hash_to_node(self, node):
        # Get the children of node
        child_nodes = sorted(list(self.merkle_tree_stripped.successors(node)))
        if len(child_nodes) == 2:
            hash_left = self.hashes[child_nodes[0]]
            hash_right = self.hashes[child_nodes[1]]
        else:
            hash_left = self.hashes[child_nodes[0]]
            hash_right = self.hashes[child_nodes[0]]
        parent_data = hash_left + hash_right
        return calculate_hash_secret(parent_data)

    def calculate_hash_root(self):
        path = nx.shortest_path(self.merkle_tree_stripped, source=self.root_id, target=self.id)
        path.reverse()
        for node in path:
            if node == self.id:
                continue
            hash = self.calculate_hash_to_node(node)
            self.hashes[node] = hash
        return self.hashes[self.root_id]


def has_single_child(G, node):
    parents = list(G.predecessors(node))
    if len(parents) == 1:
        parent = parents[0]
        children = list(G.successors(parent))
        return len(children) == 1
    return False


def draw_merkle_tree(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True)
    plt.show()


def generate_1key_parts(key, n, start_dict_key):
    if n <= 22:
        key_length = len(key)
        part_length = key_length // n
        key_parts = {}
        start = 0
        for i in range(n):
            if i == n - 1:
                key_parts[i + start_dict_key] = key[start:]
            else:
                end = start + part_length
                key_parts[i + start_dict_key] = key[start:end]
                start = end
        return key_parts
    else:
        return False


def generate_key_parts(n):
    key = Fernet.generate_key()
    if n <= 22:
        key_parts = generate_1key_parts(key, n, 0)
        grouping = list(key_parts.keys())
    else:
        key_num = int(math.ceil(n / 22))
        key_parts = {}
        grouping = []
        for i_key in range(key_num):
            parts_num = n - i_key * 22
            if parts_num > 22:
                parts_num = 22
            key_parts_new = generate_1key_parts(key, parts_num, i_key * 22)
            key_parts.update(key_parts_new)
            grouping.append(list(key_parts_new.keys()))
    return key, key_parts, grouping


def combine_key_parts(key_parts):
    key = b''
    key_parts_sorted = dict(sorted(key_parts.items()))
    for part in key_parts_sorted.values():
        key += part
        if len(key) >= 44:
            break
    return key


def generate_key():
    """
    Generates a key and save it into a file
    """
    key = Fernet.generate_key()
    with open('secret.key', 'wb') as key_file:
        key_file.write(key)


def load_key():
    """
    Load the previously generated key
    """
    return open('secret.key', 'rb').read()


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert Tensor to a nested list
        return super().default(obj)


def encrypt_dict(dict_to_encrypt, key):
    """
    Encrypt a dictionary
    """
    str_ = json.dumps(dict_to_encrypt, cls=TensorEncoder)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(str_.encode())

    return encrypted


def decrypt_dict(encrypted_dict, key):
    """
    Decrypt a dictionary
    """
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_dict)

    return json.loads(decrypted.decode())


# %%
if __name__ == '__main__':
    leaf_node_data_dict = {
        1: 'key part 1',
        2: 'key part 2',
        3: 'key part 3',
        4: 'key part 4',
        5: 'key part 5',
        6: 'key part 6',
        7: 'key part 7',
        8: 'key part 8',
        9: 'key part 9',
        10: 'key part 10',
        11: 'key part 11',
        12: 'key part 12',
        13: 'key part 13',
    }
    my_merkle_tree = MerkleTree(leaf_node_data_dict)

    draw_merkle_tree(my_merkle_tree.G)
    root = '4_1'

    for node_id in range(1, 14):
        print('------------------')
        print(f'Node id {node_id}')
        info, path = my_merkle_tree.get_hashes_to_root(node_id)
        print(f'Path to merkle root is {path}')
        print(f'Hashes needed from nodes {list(info.keys())}')
        hashes = {
            node_id: calculate_hash_secret(leaf_node_data_dict[node_id]),
        }
        hashes.update(info)
        node_secret_sharing = SecretRecipient(node_id, root, my_merkle_tree.G, hashes)
        hash, flag = node_secret_sharing.calculate_hash_root()
        print(f'Hash of root is {hash}')
        print(f'Same hash as root? {flag}')

    # Generate key parts
    n = 30
    key, key_parts, grouping = generate_key_parts(n)

    key_parts1 = {key: key_parts[key] for key in grouping[0]}
    key_parts2 = {key: key_parts[key] for key in grouping[1]}

    # Combine key parts to obtain the full key
    key1 = combine_key_parts(key_parts1)
    key2 = combine_key_parts(key_parts2)

    dict_to_encrypt = {"key1": "value1", "key2": "value2"}

    # encrypt the dictionary
    encrypted = encrypt_dict(dict_to_encrypt, key1)
    print(encrypted)  # it's bytes, not a dictionary anymore

    # decrypt the dictionary
    decrypted = decrypt_dict(encrypted, key1)
    print(decrypted)  # it's a dictionary
