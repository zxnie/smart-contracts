import hashlib
from cryptography.fernet import Fernet
import json
import torch


class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert Tensor to a nested list
        return super().default(obj)


def generate_key():
    return Fernet.generate_key()


def encrypt_dict(dict_to_encrypt, key):
    '''
    Encrypt a model
    :param dict_to_encrypt:
    :param key:
    :return:
    '''

    str_ = json.dumps(dict_to_encrypt, cls=TensorEncoder)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(str_.encode())

    return encrypted


def decrypt_dict(encrypted_dict, key):
    '''
    Decrypt a model
    '''
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_dict)

    return json.loads(decrypted.decode())


def calculate_hash_secret(data):
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha256(data.encode('utf-8')).hexdigest()
