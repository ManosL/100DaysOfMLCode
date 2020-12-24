import gym
import random

# This is a program that will solve cart pole
# using the Monte Carlo
# with linear approximation
# algorithm and the features will be the observation
# vector + the action num

def zeros_2d(x_len, y_len):
    arr = []

    for _ in range(x_len):
        arr.append([0] * y_len)

    return arr

def dot_product(vec1, vec2):
    val = 0

    assert len(vec1) == len(vec2)

    for i in range(len(vec1)):
        val += vec1[i] * vec2[i]

    return val

def argmax(vec):
    max_val   = None
    max_index = None

    for i in range(len(vec)):
        if (max_index == None) or (vec[i] > max_val):
            max_val   = vec[i]
            max_index = i
    
    return max_index

def random_vec(n):
    vec = []

    for _ in range(n):
        vec.append(random.random())
    
    return vec

class CartPoleMonteCarlo:
    def __init__(self, epsilon=0.05, gamma=1, lr=0.01):
        self.__epsilon       = epsilon
        self.__gamma         = gamma
        self.__learning_rate = lr
        self.__weights       = zeros_2d(2, 4)
        self.__env           = gym.make('CartPole-v0')

    def reset(self):
        self.__env     = gym.make('CartPole-v0')
        self.__weights = zeros_2d(2, 4)

    
    def __epsilon_greedy(self, state):
        actions   = self.__env.action_space.n
        state_action_values = []

        for i in range(actions):
            features       = list(state)

            state_action_values.append(dot_product(features, self.__weights[i]))
        
        strategy = random.random()

        if strategy >= self.__epsilon:
            return argmax(state_action_values)
        else:
            return random.randint(0, actions - 1)

    def __normalize_weights(self):
        for i in range(len(self.__weights)):
            max_val = max([abs(elem) for elem in self.__weights[i]])

            if max_val == 0.0:
                continue

            for j in range(len(self.__weights[i])):
                self.__weights[i][j] /= max_val

        return

    def run(self, episodes=1000):
        for ep_num in range(episodes):
            #self.__normalize_weights()

            #if ep_num == 10000:
            #    self.__epsilon = 0.2

            if ep_num % 1000 == 0:
                self.__epsilon *= 0.9

            curr_state  = self.__env.reset()
            curr_action = self.__epsilon_greedy(curr_state)

            episode_steps   = 0
            episode_summary = [] # will hold tuples (state, reward, action)

            while 1:
                self.__env.render()
                next_state, curr_reward, done, _ = self.__env.step(curr_action)
                episode_steps += 1

                if done:
                    episode_summary = [(next_state, 0, None), (curr_state, curr_reward, curr_action)] + episode_summary

                    print("Episode", ep_num, "finished after", episode_steps, "steps")
                    break
            
                episode_summary = [(curr_state, curr_reward, curr_action)] + episode_summary
                curr_state      = next_state
                curr_action     = self.__epsilon_greedy(curr_state)
            
            print([ep_sum[2] for ep_sum in episode_summary])
            curr_goal = episode_summary[0][1]

            for i in range(1, len(episode_summary)):
                state, reward, action = episode_summary[i]

                features            = list(state)
                stt_act_val         = dot_product(features, self.__weights[action])

                curr_goal   = reward + self.__gamma * curr_goal
                error       = curr_goal - stt_act_val
                print(error)
                # Doing w = a*[G_t - q(S,A,w)]*features(S,A)
                for j in range(len(self.__weights[0])):
                    self.__weights[action][j] += self.__learning_rate * error * features[j]

        self.__env.close()

        return 

def main():
    sol = CartPoleMonteCarlo(epsilon=0.35, gamma=1, lr=0.01)
    sol.run(300000)

if __name__ == "__main__":
    main()