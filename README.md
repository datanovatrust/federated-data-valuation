# 🛡️ Federated Data Valuation: Data Valuation and Detection in Federated Learning

![Federated Data Valuation Banner](banner.svg)

Welcome to **Federated Data Valuation**, a federated learning framework that implements the methods proposed in the paper:

> **Data Valuation and Detections in Federated Learning**  
> Wenqian Li, Shuran Fu, Fengrui Zhang, Yan Pang  
> [arXiv:2311.05304](https://arxiv.org/abs/2311.05304)

This project leverages Wasserstein distance to evaluate client contributions and detect noisy or irrelevant data in a privacy-preserving manner. It provides a scalable and efficient solution for data valuation in federated learning without relying on validation datasets.

---

## Table of Contents

- [🌟 Features](#-features)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [🛠️ Usage](#️-usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [🏦 Blockchain Integration for DP FL](#-blockchain-integration-for-dp-fl)
  - [Installing Brownie](#installing-brownie)
  - [Installing and Running Anvil (Foundry)](#installing-and-running-anvil-foundry)
  - [Compiling and Deploying the Smart Contract](#compiling-and-deploying-the-smart-contract)
  - [Integrating the Deployed Contract](#integrating-the-deployed-contract)
  - [Running Federated Training with Blockchain Integration](#running-federated-training-with-blockchain-integration)
  - [Citation for Privacy-Preserving Blockchain-Enabled FL](#citation-for-privacy-preserving-blockchain-enabled-fl)
- [📁 Project Structure](#-project-structure)
- [🔍 Detailed Description](#-detailed-description)
  - [Federated Learning Workflow](#federated-learning-workflow)
  - [Client Contribution Evaluation](#client-contribution-evaluation)
  - [Data Detection](#data-detection)
- [📊 Results](#-results)
- [📝 Notes](#-notes)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👩‍💻 Authors](#-authors)
- [🙏 Acknowledgements](#-acknowledgements)
- [📞 Contact](#-contact)
- [📚 References](#-references)

---

## 🌟 Features

- **Privacy-Preserving Data Valuation**: Evaluate client contributions using Wasserstein distance without sharing raw data.
- **No Validation Dataset Required**: Efficient computation of Wasserstein barycenter eliminates the need for validation datasets.
- **Data Detection**: Identify and filter out noisy or irrelevant data points.
- **Scalability**: Optimized for large-scale federated learning with numerous clients.
- **Advanced Models**: Supports Vision Transformer (ViT) and ResNet architectures.
- **Optimized Training and Evaluation**: Utilizes multiprocessing and efficient data loading techniques.
- **Logging and Visualization**: Detailed logging and plots for client contributions and training accuracy.
- **Extensible**: Easy to integrate with custom datasets and models.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.11.8 (has not been tested with other versions of Python)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
- [transformers](https://huggingface.co/transformers/)
- Other dependencies listed in `requirements.txt`
- Or, simply run the `setup.sh` file, which will set up a new conda environment for you and install the necessary requirements

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/datanovatrust/federated-data-valuation.git
   cd federated-data-valuation
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage

### Configuration

Before running the training script, you can adjust the configurations in `src/config/config.yaml`:

```yaml
model:
  name: vit  # Options: 'vit', 'resnet'
  num_labels: 10

training:
  batch_size: 32
  epochs: 1
  learning_rate: 1e-4

federated_learning:
  num_clients: 5
  rounds: 5
  fraction_fit: 0.6  # Fraction of clients selected
  num_shards: 50      # Partitions data into 50 shards
  num_samples: 100    # Limits dataset size to 100 samples

fedbary:
  use_validation: true  # Use validation data for global distribution
```

### Training

To start federated training:

```bash
python scripts/train_federated.py
```

You can also specify the differential privacy epsilon value via command-line argument:

```bash
python scripts/train_federated.py --epsilon=10
```

### Evaluation

After training, evaluation metrics and plots are saved in the `experiments` directory:

- **Client Contributions Plot**: `experiments/client_contributions.png`
- **Training Accuracy Plot**: `experiments/training_accuracy.png`
- **Confusion Matrix**: `experiments/confusion_matrix_round_{round_num}.png` (One for each round)

---

## 🏦 Blockchain Integration for DP FL

To enhance privacy and security, we integrate Blockchain and Differential Privacy into the federated learning pipeline. This approach records global model updates on-chain, leverages IPFS for decentralized storage, and can use differential privacy to protect client data. By default, we use a local Ethereum node (Anvil from Foundry) and a locally running IPFS node to avoid authentication issues with Infura.

### Prerequisites for the Blockchain Setup

1. **Ethereum Node (Anvil)**:  
   We use Anvil (part of Foundry) as the local Ethereum node.

2. **Brownie**:  
   A Python framework for Ethereum. Used to compile, deploy, and interact with smart contracts.

3. **IPFS Node (Local)**:  
   We run a local IPFS daemon to store global model checkpoints without relying on external services.

### Installing Brownie

Ensure you are in the `federated-data-valuation` directory and have the virtual environment activated:

```bash
pip install eth-brownie
```

Check Brownie installation:

```bash
brownie --version
```

You should see a Brownie version output.

### Installing and Running Anvil (Foundry)

1. **Install Foundry & Anvil**:

   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   # Follow the instructions, then:
   source ~/.zshenv
   foundryup
   ```

2. **Run Anvil**:

   In a separate terminal:

   ```bash
   anvil
   ```

   Keep anvil running in this terminal. It listens at `http://127.0.0.1:8545`.

### Compiling and Deploying the Smart Contract

1. **Compile the Contract**

   We assume you have `FLRegistry.sol` in `contracts/`. Run:

   ```bash
   brownie compile
   ```

2. **Deploy the Contract**

   Use the provided deployment script (e.g., `scripts/deploy_contract.py`):

   ```bash
   brownie run scripts/deploy_contract.py --network development
   ```

   Brownie connects to anvil. It prints the deployed contract address. Copy that address.

### Integrating the Deployed Contract

1. **Extract the ABI**

   After compilation, your contract ABI is in `build/contracts/FLRegistry.json`. Extract the ABI:

   ```bash
   cat build/contracts/FLRegistry.json | jq '.abi' > src/config/FLRegistry_abi.json
   ```

2. **Update `blockchain_config.yaml`**

   Edit `src/config/blockchain_config.yaml` to point to your deployed contract, local RPC, and local IPFS:

   ```yaml
   blockchain:
     enabled: true
     rpc_url: "http://127.0.0.1:8545"    # Anvil endpoint
     contract_address: "0xYourDeployedContractAddress"
     ipfs_gateway: "http://127.0.0.1:5001" # Local IPFS daemon endpoint
     abi_file: "src/config/FLRegistry_abi.json"
   ```

### Installing and Running IPFS Locally

1. **Install IPFS**:

   On macOS (Homebrew):
   ```bash
   brew install ipfs
   ```

   On Linux, follow instructions from:
   [https://docs.ipfs.tech/install/](https://docs.ipfs.tech/install/)

2. **Initialize and Run IPFS Daemon**:

   ```bash
   ipfs init
   ipfs daemon
   ```

   This runs IPFS locally with an API at `127.0.0.1:5001`.

### Running Federated Training with Blockchain and DP

With Anvil running, contract deployed, IPFS daemon running, and `blockchain_config.yaml` pointing to local IPFS and Ethereum node:

```bash
python scripts/train_blockchain_federated.py
```

**What this does:**

- Performs federated training as in `train_federated.py`.
- Uses DP (if `--epsilon` is provided) for privacy.
- Uploads global model checkpoints to IPFS (no authentication required locally).
- Records the model hash on the Blockchain contract.
- If a `PRIVATE_KEY` environment variable is set, it can send transactions to record model updates on-chain. Without a private key, it will only read from the Blockchain.

### Troubleshooting

- **IPFS Connection Refused**: Make sure `ipfs daemon` is running locally.
- **Blockchain Issues**: Ensure Anvil is running and contract is deployed.

### Citation for Privacy-Preserving Blockchain-Enabled FL

This project also references privacy-preserving approaches in Blockchain-based Federated Learning as discussed in:

```
Privacy-Preserving in Blockchain-based Federated Learning Systems
Sameera K. M., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Rafidha Rehiman K. A., Vinod P., Mauro Conti
arXiv:2401.03552
```

If you are using Blockchain and DP features, consider citing:

```
@misc{sameera2024privacy,
  title={Privacy-preserving in Blockchain-based Federated Learning Systems},
  author={K. M., Sameera and Nicolazzo, Serena and Arazzi, Marco and Nocera, Antonino and Rehiman K. A., Rafidha and P., Vinod and Conti, Mauro},
  year={2024},
  eprint={2401.03552},
  archivePrefix={arXiv},
  note={Computer Communications Journal, 2024}
}
```

---

## 📁 Project Structure

```bash
federated-data-valuation
├── Dockerfile
├── README.md
├── banner.svg
├── checkpoints
│   └── ...
├── data
│   └── ...
├── docs
├── experiments
│   └── ...
├── logs
│   └── ...
├── notebooks
│   └── federated_training.ipynb
├── requirements.txt
├── scripts
│   ├── deploy_contract.py
│   ├── run_rmia_attack.py
│   ├── train_federated.py
│   └── train_peft_federated.py
├── src
│   ├── attacks
│   ├── config
│   │   ├── config.yaml
│   │   ├── peft_config.yaml
│   │   └── blockchain_config.yaml
│   ├── models
│   ├── trainers
│   └── utils
│       ├── blockchain_utils.py
│       ├── ...
└── tests
    └── ...
```

---

## 🔍 Detailed Description

### Federated Learning Workflow

1. **Data Loading**: MNIST dataset is loaded and transformed to match the input requirements of ViT.
2. **Data Partitioning**: Data is partitioned among clients in a non-IID fashion using shards.
3. **Client Setup**: Local clients are initialized with their respective datasets.
4. **Client Contribution Evaluation**: Wasserstein distance is computed to evaluate data distribution similarity.
5. **Client Selection**: Clients are selected based on their Wasserstein distances.
6. **Training Rounds**: For each round:
   - Selected clients train the model locally.
   - Models are aggregated to update the global model.
   - The global model is evaluated on the test dataset.
7. **Results Visualization**: Plots for client contributions and training accuracy are generated.

### Client Contribution Evaluation

- **Wasserstein Distance**: Measures the distribution similarity between client data and the global distribution.
- **Federated Barycenter Computation**: Approximates the Wasserstein barycenter among client distributions.
- **Privacy Preservation**: No raw data is shared; only interpolating measures are communicated.
- **Client Selection Strategy**: Clients with the smallest Wasserstein distances are selected for training.

### Data Detection

- **Duality Theorem**: Utilizes the dual formulation of the Wasserstein distance to compute calibrated gradients.
- **Datum Evaluation**: Calculates the contribution of individual data points to the overall distance.
- **Noisy Data Detection**: Identifies and filters out noisy or irrelevant data points before training.
- **Efficiency**: Detects data issues without the need for model training or validation datasets.

---

## 📊 Results

- **Client Contributions**:

  ![Client Contributions](experiments/client_contributions.png)

- **Training Accuracy Over Rounds**:

  ![Training Accuracy](experiments/training_accuracy.png)

---

## 📝 Notes

- **Device Selection**: The script automatically uses GPU if available.
- **Optimizations**: Data loading and evaluation are optimized for performance.
- **Error Handling**: Extensive error handling and logging are implemented for robustness.
- **Scalability**: Designed to handle large numbers of clients efficiently.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Authors

- **David Zagardo** - *Initial work* - [dzagardo](https://github.com/dzagardo)

---

## 🙏 Acknowledgements

This project implements methods from the paper:

- **Data Valuation and Detections in Federated Learning**  
  Wenqian Li, Shuran Fu, Fengrui Zhang, Yan Pang  
  [arXiv:2311.05304](https://arxiv.org/abs/2311.05304)

We thank the authors for their valuable contributions to the field.

Additionally, we acknowledge the use of the **Fast Differential Privacy** (**fastDP**) library developed by Zhiqi Bu and colleagues, which provides efficient differentially private optimization for PyTorch models.

- **Fast Differential Privacy Library**  
  [GitHub Repository](https://github.com/awslabs/fast-differential-privacy)

Please consider citing their work:

```
@inproceedings{bu2023differentially,
  title={Differentially private optimization on large model at small cost},
  author={Bu, Zhiqi and Wang, Yu-Xiang and Zha, Sheng and Karypis, George},
  booktitle={International Conference on Machine Learning},
  pages={3192--3218},
  year={2023},
  organization={PMLR}
}
```

We are grateful for their valuable contributions to the field and for making their library available.

---

## 📞 Contact

Feel free to reach out for any inquiries or support.

- Email: dave@greenwillowstudios.com
- GitHub: [dzagardo](https://github.com/dzagardo)

---

## 📚 References

- Li et al., "Data Valuation and Detections in Federated Learning," [arXiv:2311.05304](https://arxiv.org/abs/2311.05304).
- Sameera K. M. et al., "Privacy-preserving in Blockchain-based Federated Learning Systems," [arXiv:2401.03552](https://arxiv.org/abs/2401.03552).
- Bu et al., "Differentially private optimization on large model at small cost," ICML 2023.
- [fastDP](https://github.com/awslabs/fast-differential-privacy)
- [PyTorch Docs](https://pytorch.org/docs/stable/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Federated Learning Concepts](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

---

# 🚀 Additional Features!

Just when you thought it couldn't get better, we've added more features! 🎉

## 🧠 Support for Custom Datasets

You can now use your own custom datasets by modifying the `load_custom_dataset` function in `src/utils/data_loader.py`. The function supports loading images and CSV files.

### Usage

In `train_federated.py`, replace the MNIST data loading with your custom dataset:

```python
from src.utils.data_loader import load_custom_dataset

train_dataset = load_custom_dataset(data_dir='path/to/your/data', file_type='jpg', transform=transform)
```

## 🌐 Extended Model Support

We've extended support for more models:

- **MobileNetV2**
- **DenseNet**

You can specify the model in the configuration

```yaml
model:
  name: mobilenet  # Options: 'vit', 'resnet', 'mobilenet', 'densenet'
  num_labels: 10
```

## 🔒 Enhanced Security

- **Data Privacy**: Implemented differential privacy mechanisms to ensure client data remains secure.
- **Secure Aggregation**: Models are aggregated using secure protocols to prevent leakage.

## 📈 Advanced Metrics

- **Confusion Matrix**: Generate confusion matrices to analyze model performance.
- **Per-Class Accuracy**: Evaluate accuracy for each class individually.

### Confusion Matrix Example

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# After model evaluation
cm = confusion_matrix(all_targets, all_predictions)
sns.heatmap(cm, annot=True)
plt.savefig('experiments/confusion_matrix.png')
```

![Confusion Matrix](experiments/confusion_matrix_round_5.png)

## 🌟 Live Monitoring Dashboard

We've integrated a live monitoring dashboard using **TensorBoard** to visualize training progress in real-time.

### How to Use

1. **Start TensorBoard**

   ```bash
   tensorboard --logdir=runs
   ```

2. **Access Dashboard**

   Open [http://localhost:6006](http://localhost:6006) in your browser.

---

I hope you enjoy these new features! If you have any suggestions or encounter any issues, please let me know. Happy coding! 💻🎉