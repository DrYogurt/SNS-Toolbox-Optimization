import torch
import gymnasium as gym
from clone import Actor
import numpy as np

# Create a random environment
env = gym.make('CartPoleContinuous-v1',render_mode='rgb_array')
# Wrap the environment with the Monitor to record a video
env = gym.wrappers.RecordVideo(env, './SNS-Toolbox-Optimization/cloned_cartpole/video/sns_cont_trained_3',episode_trigger=lambda t:True)
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.NormalizeObservation(env)
env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
env = gym.wrappers.NormalizeReward(env)
env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

# Load your trained model
input_size = env.observation_space.shape[0]
hidden_size = 2
output_size = env.action_space.shape[0]
model = Actor(input_size,hidden_size,output_size)
model.load_state_dict(torch.load('SNS-Toolbox-Optimization/cloned_cartpole/sns_trained.pth'))
model.eval()


# Set the random seed for reproducibility

# Set the maximum number of steps for the episode
max_steps = 500

# Reset the environment
observation,_ = env.reset()

# Run the model in the environment
for step in range(max_steps):
    # Convert the observation into a torch tensor
    x = torch.tensor(observation, dtype=torch.float32)

    # Perform a forward pass through the model
    output, _ = model(x, hidden=None)

    # Convert the output into a probability distribution
    #probs = torch.softmax(output, dim=-1)
    
    # Choose the action with the highest probability
    #action = probs.squeeze().argmax().item()

    # Take the chosen action in the environment
    observation, _, done, _,_ = env.step(output.detach().numpy()[0])

    if done:
        break

# Close the environment
env.close()
