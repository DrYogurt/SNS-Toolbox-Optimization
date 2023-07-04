import wandb
import os
import subprocess

# Initialize wandb with your project name and entity name
wandb.login()

sweep_config = {
    'method': 'grid',
    'metric': {
      'name': 'value_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3, 3e-4, 1e-4]
        },
        'num_envs': {
            'values': [1, 2, 4]
        },
        'num_steps': {
            'values': [1024, 2048, 4096]
        },
        'gamma': {
            'values': [0.99, 0.995]
        },
        'gae_lambda': {
            'values': [0.9, 0.95, 0.98]
        },
        'num_minibatches': {
            'values': [16, 32, 64]
        },
        'ent_coef': {
            'values': [0.01, 0.02]
        },
        'vf_coef': {
            'values': [0.0, 0.5, 1.0]
        },
        'max_grad_norm': {
            'values': [0.5, 1.0]
        },
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'total_timesteps': {
            'values': [200000, 500000, 1000000]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="behavior-cloning-feed-forward-models")

def train():
    config_defaults = {
        # Default parameters remain unchanged...
    }
    wandb.init(config=config_defaults)
    
    config = wandb.config
    # Call your training script here
    # This is just an example. Modify this according to your script name and args
    cmd = [
        "python", "ppo.py",
        "--learning_rate", str(config.learning_rate),
        #"--seed", str(config.seed),
        "--total_timesteps", str(config.total_timesteps),
        #"--torch_deterministic", str(config.torch_deterministic),
        #"--cuda", str(config.cuda),
        "--track", "true",
        "--capture_video", "true",
        "--num_envs", str(config.num_envs),
        "--num_steps", str(config.num_steps),
        #"--anneal_lr", str(config.anneal_lr),
        #"--gae", str(config.gae),
        "--gamma", str(config.gamma),
        "--gae_lambda", str(config.gae_lambda),
        "--num_minibatches", str(config.num_minibatches),
        #"--update_epochs", str(config.update_epochs),
        #"--norm_adv", str(config.norm_adv),
        #"--clip_coef", str(config.clip_coef),
        #"--clip_vloss", str(config.clip_vloss),
        "--ent_coef", str(config.ent_coef),
        "--vf_coef", str(config.vf_coef),
        "--max_grad_norm", str(config.max_grad_norm),
        "--hidden_size", str(config.hidden_size),
    ]
    subprocess.call(cmd)

wandb.agent(sweep_id, function=train)
