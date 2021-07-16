# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Utils to train DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import json
import logging
import os
import socket

import git
import numpy as np
import torch



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def move_to(obj, device):
    """
    Moves different objects to database.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

def git_log(folder_path: str):
    """
    Log commit info.
    """
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }

    with open(os.path.join(folder_path, "git_log.json"), "w") as f:
        json.dump(repo_infos, f, indent=4)


def init_gpu_params(params):
    """
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info("Initializing GPUs")
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ["WORLD_SIZE"])
        params.n_gpu_per_node = int(os.environ["N_GPU_NODE"])
        params.global_rank = int(os.environ["RANK"])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ["N_NODES"])
        assert params.node_id == int(os.environ["NODE_RANK"])

    # local job (single GPU)
    else:
        assert params.local_rank == -1  or  params.local_rank == 0

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1

    # summary
    PREFIX = f"--- Global rank: {params.global_rank} - "
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu and not params.deepspeed:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
        )
    elif params.deepspeed:
        # deepspeed performs its own DDP internally, and requires the program to be started with:
        # deepspeed  ./program.py
        # rather than:
        # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
        from transformers_local.deepspeed import is_deepspeed_available

        if not is_deepspeed_available():
            raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
        import deepspeed
        deepspeed.init_distributed()

        # workaround for setups like notebooks where the launcher can't be used,
        # but deepspeed requires a dist env.
        # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != params.local_rank:
            params.local_rank = env_local_rank
        params.is_master = params.node_id == 0 and params.local_rank == 0


        params.per_device_train_batch_size = params.batch_size 
        params.train_micro_batch_size_per_gpu = params.per_device_train_batch_size 
        params.warmup_steps = 0
        params.fp16_backend = 'auto'

        # - must be run very last in arg parsing, since it will use a lot of these settings.
        # - must be run before the model is created.
        from transformers_local.deepspeed import HfTrainerDeepSpeedConfig

        # will be used later by the Trainer (leave self.deepspeed unmodified in case a user relies on it not to be modified)
        params.hf_deepspeed_config = HfTrainerDeepSpeedConfig(params.deepspeed)
        params.hf_deepspeed_config.trainer_config_process(params)

        ## Needs to happen after call to HfTrainerDeepSpeedConfig
        #params.n_gpu = 1
        #params.n_nodes = 1
        #params.node_id = 0
        #params.global_rank = 0
        #params.world_size = 1
        #params.n_gpu_per_node = 1
        params.multi_gpu = False

        return torch.device("cuda", params.local_rank)



def set_seed(args):
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


""" 
SPECIFIC TO GLUE / SUPER GLUE
"""

GLUE_KEYS = {"cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"}

SUPER_GLUE_KEYS = {"boolq", "cb", "copa", "multirc", "record", "rte2", "wic", "wsc", "wsc.fixed", "axb", "axg"}
"""
TASK_TO_KEYS = {
    ### GLUE
    "glue": {
        "cola": ("sentence", ),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", ),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    },
    ### SUPERGLUE
    "super_glue": {
        "boolq": ("question", "passage"),
        "cb": ("premise", "hypothesis"),
        "copa": ("choice1", "choice2"),
        "multirc": ("paragraph", "question", "answer"),
        "record": ("passage", "query", "entities", "answers"),
        "rte": ("premise", "hypothesis"),
        "wic": ("word", "sentence1", "sentence2", "start1", "start2", "end1", "end2"),
        "wsc": ("text", "span1_index", "span2_index", "span1_text", "span2_text"),
        "wsc.fixed": ("text", "span1_index", "span2_index", "span1_text", "span2_text"),
        "axb": ("sentence1", "sentence2"),
        "axg": ("entailment", "not_entailment"),
    }
}
"""
TASK_TO_KEYS = {
    ### GLUE
    "glue": {
        "cola": ("sentence", ),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", ),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    },
    ### SUPERGLUE
    "super_glue": {
        "boolq": ("question", "passage"),
        "cb": ("premise", "hypothesis"),
        "copa": ("choice1", "choice2"),
        "multirc": ("paragraph", "question", "answer"),
        "record": ("passage", "query", "entities", "answers"),
        "rte": ("premise", "hypothesis"),
        "wic": ("word", "sentence1", "sentence2"),
        "wsc": ("text", "span1_text", "span2_text"),
        "wsc.fixed": ("text", "span1_text", "span2_text"),
        "axb": ("sentence1", "sentence2"),
        "axg": ("entailment", "not_entailment"),
    }
}

def task_to_dataset(task_name):
    if task_name in GLUE_KEYS:
        return "glue"
    elif task_name in SUPER_GLUE_KEYS:
        return "super_glue"
    else:
        raise ValueError("Unknown task, you should pick one in " + ",".join(GLUE_KEYS.union(SUPER_GLUE_KEYS)))

