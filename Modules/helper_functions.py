import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from env_economic_choice_original import EconomicChoiceEnv as EconomicChoiceEnv_full
from env_economic_choice_partial import EconomicChoiceEnv as EconomicChoiceEnv_partial


# ------------------------------------------
# Helper functions for REINFORCE algorithm
# ------------------------------------------

#################### Discount rewards ####################
def discount_rewards(rewards, gamma):
    """
    Computes discounted rewards.

    Args:
    -----------
    rewards: list
        List of rewards collected in an episode.
    gamma: float
        Discount factor (between 0 and 1).
    Returns:
    -----------
    discounted: np.array
        Discounted rewards.
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    return discounted

#################### Padding for hidden states ####################
def pad_hidden_states(hidden_states, max_steps, units):
    """
    Pads the hidden states to the desired shape.

    Args:
    -----------
    hidden_states: np.array
        Hidden states to pad.
    max_steps: int
        Maximum number of steps.
    units: int
        Number of units.

    Returns:
    -----------
    padded: np.array
        Padded hidden states.
    """
    padded = np.full((units, max_steps), np.nan)
    length = hidden_states.shape[1]
    padded[:, :length] = hidden_states
    return padded

#################### Other measurements collection ####################
def collect_other_measurements(env, done):
    """
    Collects environment-specific measurements at the end of an episode.

    Args:
    -----------
    env: Gymnasium.Env
        The environment instance.
    done: bool
        Flag indicating if the episode ended.

    Returns:
    -----------
    list or None: A list containing specific measurements for the target environment
                  if the episode ended successfully, otherwise None.
                  For EconomicChoiceEnv successful trials:
                  [juice_pair_LR, offer_pair_BA, chosen_juice_type]
                  e.g., [['A', 'B'], (3, 1), 'A'] means A was Left, B Right;
                  offer was 3 drops B vs 1 drop A; agent chose A (Left).
    """
    env_instance = env.unwrapped if hasattr(env, 'unwrapped') else env

    # --- Economic Choice Environment ---
    if isinstance(env_instance, EconomicChoiceEnv_full) or \
       isinstance(env_instance, EconomicChoiceEnv_partial):
        
        # Check for successful trial completion: if done and a decision was made (not fixation)
        if done and env_instance.chosen_action != 0:
            juice_pair = env_instance.trial_juice_LR
            offer_pair = env_instance.trial_offer_BA
            chosen_action = env_instance.chosen_action

            if chosen_action == 1:
                chosen_juice_type = juice_pair[0]
            elif chosen_action == 2:
                chosen_juice_type = juice_pair[1]
            else:
                chosen_juice_type = None

            if chosen_juice_type is not None and offer_pair is not None and juice_pair is not None:
                # Return juice assignment, offer amounts (B,A), and chosen type
                return [list(juice_pair), offer_pair, chosen_juice_type]
            else:
                return None
        else:
            # Episode did not end successfully with juice reward
            return None

    else:
        # Not the target environment
        return None

# ------------------------------------------
# Helper functions for training stages
# ------------------------------------------
 
#################### Load model ####################
def load_model(agent, obs_size, act_size, stage, ckpt_prefix):
    """
    Uploads the model and the masks from the previous stage (stage-1),
    builds the networks for the current stage, initializes the optimizers
    and restores the checkpoint.

    Args:
    -----------
    agent: tf.keras.models.Model
        Must have the following attributes
            - actor (with hidden_layers)
            - critic (with hidden_layers)
            - actor_optimizer
            - critic_optimizer
    obs_size: int
        Size of the observation dimension
    act_size: int
        Size of the action dimension
    stage: int
        Current stage number (e.g. 6)
    ckpt_prefix: str
        Prefix of the checkpoint directory (without the suffix "_<stage>")

    Returns:
    -----------
        this_ckpt_dir: str
            Path to the checkpoint directory of the current stage
    """

    # --- Directories ---
    prev_ckpt_dir = f"{ckpt_prefix}_{stage-1}"
    this_ckpt_dir = f"{ckpt_prefix}_{stage}"
    os.makedirs(this_ckpt_dir, exist_ok=True)

    # --- Building the networks ---
    actor_input_shape = (None, None, obs_size)

    # Calculate actor_hid_for_critic
    actor_hid_for_critic = obs_size
    actor_hid_for_critic = agent.actor.hidden_size
    critic_input_shape = (None, None, actor_hid_for_critic + act_size)
    agent.actor.build(actor_input_shape)
    agent.critic.build(critic_input_shape)
    print("Actor and Critic networks built.")

    # --- Ensure layers are built ---
    print("Performing dummy forward to build cells and weights for mask loading...")
    dummy_obs         = tf.zeros((1,1, obs_size), dtype=tf.float32)
    _                 = agent.actor(dummy_obs, training=False)
    dummy_critic_in   = tf.zeros((1,1, agent.critic.input_size), dtype=tf.float32)
    _                 = agent.critic(dummy_critic_in, training=False)

    # --- Load sparse masks from previous stage ---
    print(f"Loading masks from stage {stage-1}...")

    # Actor masks
    for i, layer in enumerate(agent.actor.hidden_layers):
        # 1) Kernel mask
        kp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_actor_layer{i}_kernel.npy')
        if os.path.exists(kp) and layer.kernel_constraint is not None:
            mask_k = np.load(kp)
            # If the layer has the attribute `cell`, it is a GRU (ModifiedGRU or standard GRU)
            if hasattr(layer, 'cell'):
                dtype_w_in = layer.cell.W_in.dtype
            else:
                # For Dense we use the dtype of the kernel directly
                dtype_w_in = layer.kernel.dtype
            layer.kernel_constraint.mask = tf.constant(mask_k, dtype=dtype_w_in)

        # 2) Recurrent mask (only if the layer is recurrent)
        rp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_actor_layer{i}_recur.npy')
        if os.path.exists(rp) and getattr(layer, 'recurrent_constraint', None) is not None:
            if hasattr(layer, 'cell'):
                mask_r = np.load(rp)
                layer.recurrent_constraint.mask = tf.constant(mask_r, dtype=layer.cell.W_rec.dtype)
            # If `layer` is Dense, W_rec does not exist and is skipped
            
    # Critic masks
    for i, layer in enumerate(agent.critic.hidden_layers):
        # 1) Kernel mask
        kp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_critic_layer{i}_kernel.npy')
        if os.path.exists(kp) and layer.kernel_constraint is not None:
            mask_k = np.load(kp)
            # If the layer has the attribute `cell`, it is a GRU (ModifiedGRU or standard GRU)
            if hasattr(layer, 'cell'):
                dtype_w_in = layer.cell.W_in.dtype
            else:
                dtype_w_in = layer.kernel.dtype
            layer.kernel_constraint.mask = tf.constant(mask_k, dtype=dtype_w_in)

        # 2) Recurrent mask (only if the layer is recurrent)
        rp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_critic_layer{i}_recur.npy')
        if os.path.exists(rp) and getattr(layer, 'recurrent_constraint', None) is not None:
            if hasattr(layer, 'cell'):
                mask_r = np.load(rp)
                layer.recurrent_constraint.mask = tf.constant(mask_r, dtype=layer.cell.W_rec.dtype)
            # If `layer` is Dense, W_rec does not exist and is skipped

    print("Masks loaded.")

    # --- Dummy step to initialize optimizers ---
    print("Initializing optimizers with dummy step...")

    # 1) Create dummy_obs to build the Actor
    dummy_obs = np.zeros((1, 1, obs_size), dtype=np.float32)

    # 2) Get a simulated "hidden output" from the Actor with a zero state:
    processed_dummy_input = agent.actor.input_fc(dummy_obs, training=False)
    dummy_last_hidden_output = processed_dummy_input
    for hidden_layer in agent.actor.hidden_layers:
        if 'GRU' in agent.actor.layer_type:
            dummy_last_hidden_output, _ = hidden_layer(dummy_last_hidden_output, training=False)
        elif agent.actor.layer_type == 'Dense':
             dummy_last_hidden_output = hidden_layer(dummy_last_hidden_output, training=False)
    hd0 = dummy_last_hidden_output.numpy()[0, 0, :]

    # 3) Concatenate with "curr_a" empty one-hot to simulate input to the Critic
    dummy_curr_a0 = np.zeros((act_size,), dtype=np.float32)
    dummy_curr_a0[0] = 1.0
    critic_feat0 = np.concatenate([hd0, dummy_curr_a0], axis=0)
    dummy_critic_in = critic_feat0.reshape((1, 1, -1))

    # 4) Perform a dummy forward pass through the Critic to initialize it
    with tf.GradientTape(persistent=True) as tape:
        a_out, _ = agent.actor(dummy_obs, training=True)
        c_out, _ = agent.critic(dummy_critic_in, training=True)
        loss_a   = tf.reduce_mean(tf.square(a_out))
        loss_c   = tf.reduce_mean(tf.square(c_out))
    grads_a = tape.gradient(loss_a, agent.actor.trainable_variables)
    grads_c = tape.gradient(loss_c, agent.critic.trainable_variables)
    agent.actor_optimizer.apply_gradients(zip(grads_a, agent.actor.trainable_variables))
    agent.critic_optimizer.apply_gradients(zip(grads_c, agent.critic.trainable_variables))
    del tape
    print("Optimizers initialized.")

    # --- Restore checkpoint of previous stage ---
    ckpt = tf.train.Checkpoint(
        actor=agent.actor,
        critic=agent.critic,
        actor_optimizer=agent.actor_optimizer,
        critic_optimizer=agent.critic_optimizer
    )
    manager = tf.train.CheckpointManager(ckpt, prev_ckpt_dir, max_to_keep=3)
    print(f"Restoring from checkpoint: {manager.latest_checkpoint}")
    status = ckpt.restore(manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    print("Checkpoint restored successfully.")

    return this_ckpt_dir

#################### Save model ####################
def save_model(agent, stage, ckpt_prefix):
    """
    Saves the model and its sparse masks for the current stage.

    Args:
    -----------
    agent: tf.keras.models.Model
        Must have the following attributes
            - actor (with hidden_layers)
            - critic (with hidden_layers)
            - actor_optimizer
            - critic_optimizer
    stage: int
        Current stage number (e.g. 6)
    ckpt_prefix: str
        Prefix of the checkpoints directory (without the suffix "_<stage>")

    Returns:
    -----------
    path: str
        Path to the checkpoint saved by tf.train.CheckpointManager.save()
    """
    this_ckpt_dir = f"{ckpt_prefix}_{stage}"
    os.makedirs(this_ckpt_dir, exist_ok=True)

    # --- Save checkpoint for stage n ---
    ckpt = tf.train.Checkpoint(
        actor=agent.actor,
        critic=agent.critic,
        actor_optimizer=agent.actor_optimizer,
        critic_optimizer=agent.critic_optimizer
    )
    manager = tf.train.CheckpointManager(ckpt, this_ckpt_dir, max_to_keep=3)
    path = manager.save()
    print(f"Checkpoint stage {stage} saved at: {path}")

    # --- Save sparse masks for stage n ---
    print(f"Saving masks for stage {stage}...")

    # Actor masks
    for i, layer in enumerate(agent.actor.hidden_layers):
        if layer.kernel_constraint is not None and layer.kernel_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_actor_layer{i}_kernel.npy'),
                    layer.kernel_constraint.mask.numpy())
        if getattr(layer, 'recurrent_constraint', None) is not None and layer.recurrent_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_actor_layer{i}_recur.npy'),
                    layer.recurrent_constraint.mask.numpy())
    # Critic masks (not necessary if critic is dense)
    for i, layer in enumerate(agent.critic.hidden_layers):
        if layer.kernel_constraint is not None and layer.kernel_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_critic_layer{i}_kernel.npy'),
                    layer.kernel_constraint.mask.numpy())
        if getattr(layer, 'recurrent_constraint', None) is not None and layer.recurrent_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_critic_layer{i}_recur.npy'),
                    layer.recurrent_constraint.mask.numpy())
            
    print(f"Masks saved for stage {stage}.")

    return path

# ----------------------------------
# Helper functions for plotting
# ----------------------------------

#################### Rewards plot ####################
def plot_rewards(rewards1, rewards2, window_1=100, window_2=100, labels=('Series 1', 'Series 2')):
    """
    Draws the total rewards as a function of the episodes with their moving average and their moving median.

    Args:
    -----------
    rewards1 : list or np.ndarray
        Array corresponding to the total rewards obtained in each episode of the first series.
    rewards2 : list or np.ndarray
        Array corresponding to the total rewards obtained in each episode of the second series.
    window_1 : int
        Size of the window for calculating the moving average and median of the first series.
    window_2 : int
        Size of the window for calculating the moving average and median of the second series.
    labels : tuple of two strings
        Labels for each series (default is ('Series 1', 'Series 2')).
    """
    s1 = pd.Series(rewards1)
    s2 = pd.Series(rewards2)

    mean1   = s1.rolling(window_1).mean()
    median1 = s1.rolling(window_1).median()
    mean2   = s2.rolling(window_2).mean()
    median2 = s2.rolling(window_2).median()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(rewards1, label=f"{labels[0]}", color="C0", alpha=0.2, linewidth=1)
    plt.plot(rewards2, label=f"{labels[1]}", color="C1", alpha=0.2, linewidth=1)
    plt.plot(mean1, label=f"{labels[0]} Moving average", color="C0", linewidth=1)
    plt.plot(mean2, label=f"{labels[1]} Moving average", color="C1", linewidth=1)
    plt.plot(median1, label=f"{labels[0]} Moving median", color="C0", linewidth=1, linestyle='dotted')
    plt.plot(median2, label=f"{labels[1]} Moving median", color="C1", linewidth=1, linestyle='dotted')
    plt.title("Total reward per episode", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total reward", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#################### Psychometric curves plot ####################
def plot_psychometric_curve(measurements1, measurements2, labels=('Series 1', 'Series 2'), rel_value_A=2.2):
    """
    Plots the psychometric curves for two sets of measurements.

    Args:
    -----------
    measurements1 : list
        List of measurements for the first set, each element is of the form [['B', 'A'], (b, a), choice]
    measurements2 : list
        List of measurements for the second set, each element is of the form [['B', 'A'], (b, a), choice]
    labels : tuple
        Tuple containing the labels for the two sets of measurements (default is ('Series 1', 'Series 2')).
    rel_value_A : float
        Relative value of A in units of B (default is 2.2).
    """
    
    def compute_stats(measurements):
        """
        Computes statistics from the measurements.
        
        Args:
        -----------
        measurements : list
            List of measurements, each element is of the form [['B', 'A'], (b, a), choice].

        Returns:
        --------
        stats : dict
            Dictionary with keys as offer strings and values as dictionaries containing:
            - 'B_count': count of times B was chosen
            - 'total': total count of offers
            - 'relative_reward': relative reward (reward)
        """
        
        stats = defaultdict(lambda: {'B_count': 0, 'total': 0, 'relative_reward': 0.0})
        for idx, item in enumerate(measurements):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            pair, amounts, choice = item
            if not isinstance(amounts, (list, tuple)) or len(amounts) != 2:
                continue
            b, a = amounts
            try:
                key = f"{b}B:{a}A"
                val_A = a * rel_value_A
                val_B = b
                reward = val_A / val_B if val_B != 0 else float('inf')
                stats[key]['total'] += 1
                if choice == 'B':
                    stats[key]['B_count'] += 1
                stats[key]['relative_reward'] = reward
            except Exception as e:
                print(f"Error in input {idx}: {item} → {e}")
                continue
        return stats

    # Compute stats for both measurement sets
    stats1 = compute_stats(measurements1)
    stats2 = compute_stats(measurements2)

    # Unify all offers and their relative rewards
    all_offers = {}
    for offer, data in {**stats1, **stats2}.items():
        all_offers[offer] = data['relative_reward']

    # Sort offers by relative reward descending
    sorted_offers = sorted(all_offers.items(), key=lambda x: -x[1])

    x_labels = [offer for offer, _ in sorted_offers]

    # Extract percentages for both lists
    y1, y2 = [], []
    for offer in x_labels:
        pct1 = (stats1[offer]['B_count'] / stats1[offer]['total'] * 100) if offer in stats1 else None
        pct2 = (stats2[offer]['B_count'] / stats2[offer]['total'] * 100) if offer in stats2 else None
        y1.append(pct1)
        y2.append(pct2)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_labels, y1, color='C0', marker='D', s = 200, label=labels[0], alpha=0.5)
    plt.scatter(x_labels, y2, color='C1', marker='o', s = 200, label=labels[1], alpha = 0.5)
    plt.xlabel("Offer (#B:#A)", fontsize=14)
    plt.ylabel("Percentage of B choice [%]", fontsize=14)
    plt.title("Psychometric curve. 1A = 2.2B", fontsize=18)
    plt.ylim(-5, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()