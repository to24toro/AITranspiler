import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ray
import tensorflow as tf
import yaml

from rl import game
from rl.buffer import ReplayBuffer
from rl.mcts import MCTS
from rl.network import ResNet

# Load training settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

training_settings = config["training_settings"]
network_settings = config["network_settings"]
mcts_settings = config["mcts_settings"]


@dataclass
class Sample:
    state: np.ndarray
    mcts_policy: np.ndarray
    reward: float


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, test=False):
    """Perform a self-play game and collect training data."""
    record = []
    if test:
        state = game.get_initial_test_state()
    else:
        state = game.get_initial_state()
    game.reset_used_columns()
    network = ResNet(action_space=game.ACTION_SPACE)

    # Initialize network parameters
    network.predict(game.encode_state(state))
    network.set_weights(weights)

    mcts = MCTS(network=network)
    done = False
    total_score = 0
    step_count = 0
    prev_action = None

    while not done and step_count < game.MAX_STEPS:
        mcts_policy = mcts.search(
            root_state=state,
            num_simulations=mcts_settings["num_mcts_simulations"],
            prev_action=prev_action,
        )
        action = np.random.choice(range(game.ACTION_SPACE), p=mcts_policy)
        record.append(Sample(state.copy(), mcts_policy, reward=None))
        state, done, action_score = game.step(state, action, prev_action, mcts_policy)
        prev_action = action
        total_score += action_score
        step_count += 1

    # The reward is calculated based on the final state
    reward = game.get_reward(state, total_score)

    # Assign the reward to each sample
    for sample in record:
        sample.reward = reward

    return record


def main(test=False):
    """Main training loop."""
    num_cpus = training_settings["num_cpus"]
    n_episodes = training_settings["n_episodes"]
    buffer_size = training_settings["buffer_size"]
    batch_size = training_settings["batch_size"]
    epochs_per_update = training_settings["epochs_per_update"]
    update_period = training_settings["update_period"]
    save_period = training_settings["save_period"]
    os.makedirs("checkpoints", exist_ok=True)

    ray.init(num_cpus=num_cpus, num_gpus=1, local_mode=False)

    logdir = Path(__file__).parent / "log"
    if logdir.exists():
        shutil.rmtree(logdir)
    summary_writer = tf.summary.create_file_writer(str(logdir))

    game.initialize_game()  # Initialize game variables

    network = ResNet(action_space=game.ACTION_SPACE)

    # Initialize network parameters
    dummy_state = game.encode_state(game.get_initial_state())
    network.predict(dummy_state)

    current_weights = ray.put(network.get_weights())

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=network_settings["learning_rate"]
    )

    replay = ReplayBuffer(buffer_size=buffer_size)

    # Start self-play workers
    work_in_progresses = [
        selfplay.remote(current_weights, test) for _ in range(num_cpus - 1)
    ]

    n_updates = 0
    n = 0
    while n <= n_episodes:
        for _ in range(update_period):
            # Wait for a self-play worker to finish
            finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
            replay.add_record(ray.get(finished[0]))
            # Start a new self-play worker
            work_in_progresses.extend([selfplay.remote(current_weights, test)])
            n += 1

        # Update network
        if len(replay) >= batch_size:
            num_iters = epochs_per_update * (len(replay) // batch_size)
            for i in range(num_iters):
                states, mcts_policy, rewards = replay.get_minibatch(
                    batch_size=batch_size
                )
                with tf.GradientTape() as tape:
                    p_pred, v_pred = network(states, training=True)
                    value_loss = tf.square(rewards - v_pred)
                    policy_loss = -tf.reduce_sum(
                        mcts_policy * tf.math.log(p_pred + 1e-5), axis=1, keepdims=True
                    )
                    loss = tf.reduce_mean(value_loss + policy_loss)
                grads = tape.gradient(loss, network.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optimizer.apply_gradients(zip(grads, network.trainable_variables))
                n_updates += 1

                if i % 100 == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar(
                            "value_loss", tf.reduce_mean(value_loss), step=n_updates
                        )
                        tf.summary.scalar(
                            "policy_loss", tf.reduce_mean(policy_loss), step=n_updates
                        )

            current_weights = ray.put(network.get_weights())

        if n % save_period == 0:
            network.save_weights(f"checkpoints/network_{n}.weights.h5")


if __name__ == "__main__":
    main()
