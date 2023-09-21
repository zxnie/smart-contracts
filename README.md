# Smart Contract Testbed for Cyber-Physical Multi-Robot Systems in Smart Factories

This repository offers a testbed for smart contracts designed to maintain operational integrity in a cyber-physical multi-robot system set within a smart factory practicing federated learning.

## Highlights
- **Multiple Controllers & Clients**: The architecture supports multiple edge servers (controllers) and multiple robotic agents (clients).

- **Cyber-Physical Agents**: Each robot is a cyber-physical agent. Data is harvested as the robot interacts with raw materials (specifically, wafers).

- **Scalable DPoW Consensus**: Uses a modified Scalable Delegated Proof-of-Work (DPoW) consensus to suit multiple controllers.

- **Non-Stationary Environment**: Supports dataset refreshments. Assumed batch variation exists in the raw materials.

- **Smart Contract & Data Integrity**:
    - Enforces the use of key hashes as immutable transactions.
    - Implements key hash verification for guaranteed data integrity.

- **Wireless Communication Tracking**: Utilizes `simpy` to keep track of the wireless communications between robots and controllers.

## Getting Started
- Dependencies are listed in pyproject.toml.
- Main file is lib_simulation_v5.py, wrapped as object and how to use it is shown in the main function.
- The blockchain will be saved as a json file, and federated learning results and wireless results will be saved as a pkl file, an example is given.

## Resources
- Blockchain code is adapted from: [section.io - How to create a blockchain in Python](https://www.section.io/engineering-education/how-to-create-a-blockchain-in-python/)
- Dataset is sourced from: [WaferMap by Junliangwangdhu](https://github.com/Junliangwangdhu/WaferMap)

## Note
- This repository is stripped from a larger project subject to patent application. The patent link will be updated here once it is published.