import torch
import gymnasium as gym
from clone import Actor

# Create a random environment
env = gym.make('CartPole-v1',render_mode='rgb_array')
# Wrap the environment with the Monitor to record a video
env = gym.wrappers.RecordVideo(env, './video/sns_trained',episode_trigger=lambda t:True)

# Load your trained model
input_size = env.observation_space.shape[0]
hidden_size = 2
output_size = env.action_space.n
model = Actor(input_size,hidden_size,output_size)
model.load_state_dict(torch.load('sns_trained.pth'))
model.eval()


# Set the random seed for reproducibility
torch.manual_seed(0)

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
    probs = torch.softmax(output, dim=-1)
    
    # Choose the action with the highest probability
    action = probs.squeeze().argmax().item()

    # Take the chosen action in the environment
    observation, _, done, _,_ = env.step(action)

    if done:
        break

# Close the environment
env.close()
