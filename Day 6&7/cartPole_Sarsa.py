import gym
import random

# This is a program that will solve cart pole
# using the Semi Gradient Sarsa(Sutton & Barto pg 244) 
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

class CartPoleLinearSemiGradientSarsa:
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

    # I will round my features to some decimals
    def __round_features(self, features):
        for i in range(len(features)):
            features[i] = round(features[i], 2)

        return features

    def run(self, episodes=1000):
        for ep_num in range(episodes):
            #if ep_num % 100 == 0:
            #    self.__epsilon *= 0.9

            curr_state  = self.__env.reset()
            curr_action = self.__epsilon_greedy(curr_state)

            episode_steps = 0
            ep_summary    = [] 
            while 1:
                ep_summary.append(curr_action)
                self.__env.render()
                next_state, curr_reward, done, _ = self.__env.step(curr_action)
                episode_steps += 1

                if done:
                    # Doing w = w + a*[R - q(S,A,w)]*features(S,A)
                    # because derivative_w(f*w) = f
                    features    = self.__round_features(list(curr_state))

                    stt_act_val = dot_product(features, self.__weights[curr_action])
                    error       = curr_reward - stt_act_val
              
                    for i in range(len(self.__weights[0])):
                        self.__weights[curr_action][i] += self.__learning_rate * error * features[i]
                    
                    print(ep_summary)
                    print("Episode", ep_num, "finished after", episode_steps, "steps")
                    break
                
                next_action     = self.__epsilon_greedy(next_state)
                
                next_features   = self.__round_features(list(next_state))

                nxt_stt_act_val = dot_product(next_features, self.__weights[next_action])

                features        = self.__round_features(list(curr_state))

                stt_act_val     = dot_product(features, self.__weights[curr_action])

                error           = curr_reward + self.__gamma * nxt_stt_act_val
                error           = error - stt_act_val

                # Doing w = a*[R + gamma * q(S',A',w) - q(S,A,w)]*features(S,A)
                for i in range(len(self.__weights[0])):
                    self.__weights[curr_action][i] += self.__learning_rate * error * features[i]
                
                curr_state  = next_state
                curr_action = next_action

        self.__env.close()

        return 

def main():
    sol = CartPoleLinearSemiGradientSarsa(epsilon=0.05, gamma=1, lr=1)
    sol.run(30000)

if __name__ == "__main__":
    main()