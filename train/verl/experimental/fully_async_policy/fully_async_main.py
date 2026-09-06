# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


def _preseed_fully_async_actor_optim_total_training_steps(config, tokenizer, processor) -> None:
    """Megatron builds OptimizerParamScheduler during actor init and requires lr_decay_steps > 0.

    FullyAsyncTrainer initializes workers before the rollouter computes ``total_train_steps``, so
    ``actor.optim.total_training_steps`` would stay at default (-1) unless we set it here using the
    same rollout/step accounting as ``FullyAsyncRollouter.set_max_required_samples``.
    """
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.utils.dataset.rl_dataset import collate_fn

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        max_samples=config.data.get("train_max_samples", -1),
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
        num_workers=config.data["dataloader_num_workers"],
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    total_rollout_steps = len(train_dataloader) * config.trainer.total_epochs
    if config.rollout.total_rollout_steps is not None:
        total_rollout_steps = min(config.rollout.total_rollout_steps, total_rollout_steps)

    require_batches = config.async_training.require_batches
    required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * require_batches
    trigger = config.async_training.trigger_parameter_sync_step
    denom = required_samples * trigger
    total_train_steps = int(total_rollout_steps / denom)
    if total_train_steps <= 0:
        raise ValueError(
            f"[FullyAsync] Megatron LR schedule needs total_train_steps > 0, got {total_train_steps} "
            f"(total_rollout_steps={total_rollout_steps}, "
            f"ppo_mini_batch_size*require_batches*trigger_parameter_sync_step={denom}). "
            f"Increase rollout.total_rollout_steps or lower ppo_mini_batch_size, "
            f"async_training.require_batches, or trigger_parameter_sync_step."
        )

    print(
        f"[ASYNC MAIN] Pre-seeding optim.total_training_steps={total_train_steps} for Megatron "
        f"(rollout_steps={total_rollout_steps}, divisor={denom})"
    )
    try:
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            if OmegaConf.select(config, "actor_rollout_ref.actor.optim"):
                config.actor_rollout_ref.actor.optim.total_training_steps = total_train_steps
            if OmegaConf.select(config, "critic.optim"):
                config.critic.optim.total_training_steps = total_train_steps
    except Exception as e:
        print(f"Warning: Could not pre-seed total_training_steps in config. Error: {e}")


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        _preseed_fully_async_actor_optim_total_training_steps(config, tokenizer, processor)

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        from concurrent.futures import ThreadPoolExecutor

        print("[ASYNC MAIN] Creating FullyAsyncRollouter and FullyAsyncTrainer in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Rollouter does not permit continuous allocation, so we allocate trainer first.
            trainer_future = executor.submit(self._create_trainer, config)
            trainer_future.result()

            rollouter_future = executor.submit(self._create_rollouter, config)
            rollouter_future.result()

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Setting up parameter synchronization...")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    assert config.async_training.use_trainer_do_validate is False, "use_trainer_do_validate is not ready to use."

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
