import gym

# This is a program that gets the basic info
# after each step in order to provide it to
# RL algorithms

env = gym.make('CartPole-v0')

for episode in range(1000):
	observation = env.reset()
	steps = 0
	action = 1

	while 1:
		env.render() # Renders the current state of the environment

		action = env.action_space.sample()

		# observation = next state
		# reward = reward of the action I did
		# If next state is terminal(so the episode finished)
		# info for debugging (this should never used for learning)
		observation, reward, done, info = env.step(action)

		steps += 1
		if done:
			print('Episode steps =', steps)
			env.reset()
			break

env.close()
