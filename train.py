"""going by o2f format and using huggingface library
15.09.2020 - @yashbonde"""

import os
from types import SimpleNamespace
from argparse import ArgumentParser
# from data import StdzDataset, DatasetConfig, load_tokenizer

from data import DatasetConfig, WeatherDataset
from heirarchical_transformers import HeirarchicalTransformer, HeirarchicalTransformerConfig

# from transformers import GPT2Config, GPT2LMHeadModel

# --- arguments
args = ArgumentParser(description="GPT based standardisation methods")

# --- paths
args.add_argument("--save_folder", default = "models", type = str, help = "folder to save all models")
args.add_argument("--name", type = str, help = "name of this particular model")
args.add_argument("--hdf_path", default = "/Users/yashbonde/Desktop/AI/vv2/_notebooks/weatherGiga2.hdf5",
    type = str, help = "path to hdf5 dump file")
args.add_argument("--train_index", default = "/Users/yashbonde/Desktop/AI/vv2/_notebooks/test_idx.txt",
    type = str, help = "path to file with training index")
args.add_argument("--test_index", default = "/Users/yashbonde/Desktop/AI/vv2/_notebooks/test_idx.txt",
    type = str, help = "path to file with testing index")

args.add_argument("--seed", default = None, type = int, help = "seed value for training")

# --- arch
args.add_argument("--n_node_embd", default=144, type=int, help="Emebdding dim for each ndoe")
args.add_argument("--n_global_embd", default=144, type=int, help="Embedding dim for global vector")
args.add_argument("--n_layer", default = 12, type = int, help = "Num Layers")
args.add_argument("--n_head", default = 6, type = int, help = "Num Heads")
args.add_argument("--maxlen", default = 200, type = int, help = "Maximum length of decoder")
args.add_argument("--n_input_features", default=17, type=int, help="Number of weather features used")
args.add_argument("--n_location_features", default=3, type=int, help="Number of features against each location ")

# --- trainer
args.add_argument("--n_epochs", default = 200, type = int, help = "Number of epochs to train")
args.add_argument("--batch_size", default = 200, type = int, help = "Mini-Batch Size")
args.add_argument("--lr", default = 1e-3, type = float, help = "Learning Rate")
args.add_argument("--sample_every", default = 5, type = int, help = "After this epochs, create a sample dump")
args.add_argument("--train_ratio", default = 0.9, type = float, help = "Ratio of train data, rest is testing")
args.add_argument("--beta1", default = 0.9, type = float, help = "Adam.beta1")
args.add_argument("--beta2", default = 0.95, type = float, help = "Adam.beta2")
args.add_argument("--grad_norm_clip", default = 1.0, type = float, help = "Adam.beta2")

args.add_argument("--patience", default = 6, type = int, help = "training stops after patience runs out")

# --- parse and add more
args = args.parse_args()
tb_path = os.path.join(args.save_folder, args.name)
ckpt_path = os.path.join(tb_path, f"{args.name}.pt")
args = SimpleNamespace(**vars(args), ckpt_path = ckpt_path, tb_path = tb_path)

# make folders
os.makedirs(args.save_folder, exist_ok=True)
os.makedirs(args.tb_path, exist_ok=False)

# DataSet
dataTrainConfig = DatasetConfig(
    maxlen=args.maxlen,
    hdf_fpath=args.hdf_fpath,
    index=args.train_index
)
dataTestConfig = DatasetConfig(
    maxlen=args.maxlen,
    hdf_fpath=args.hdf_fpath,
    index=args.test_index
)

dtrain = WeatherDataset(dataTrainConfig, mode="train")
dtest = WeatherDataset(dataTestConfig, mode="test")

# Model
modelConfig = HeirarchicalTransformerConfig(
    n_embd=args.n_node_embd,
    n_global=args.n_global_embd,
    maxlen=args.maxlen,
    n_head=args.n_head,
    n_layer=args.n_layer,
    num_nodes=dtrain.edge_mat.shape[0],
    input_dim=args.n_input_features,
    location_dim=args.n_location_features
)

model = HeirarchicalTransformer(modelConfig)
print(f"Number of parameters: {model.num_parameters()}")

# Trainer
trainConfig = TrainerConfig(
    max_epochs = args.n_epochs,
    batch_size = args.batch_size,
    lr = args.lr,
    betas = (args.beta1, args.beta2),
    grad_norm_clip = args.grad_norm_clip,
    tb_path = args.tb_path,
    ckpt_path = args.ckpt_path,
    num_batch = (len(dtrain) // args.batch_size) + int(len(dtrain) % args.batch_size != 0),
    len_data = len(dtrain)
)

print(modelConfig, dataTrainConfig, dataTestConfig, trainConfig)
trainer = Trainer(model, dtrain, dtest, trainConfig, t)
trainer.train()
