import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        # TODO: Implement loading Q values from pickle file.
        with open(filename + '.pickle', 'rb') as file:
            self.q = pickle.load(file)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        with open(filename + '.pickle', 'wb') as file:
            pickle.dump(self.q, file)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        action = None

        # Exploration
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            if return_q:
                return (action, 0.0)
            else:
                return action
        else: # Exploitation
            
            max_q = max([self.getQ(state,a) for a in self.actions])
            list_of_best_actions = [action for action in self.actions if self.getQ(state,action) == max_q]

            best_action = list_of_best_actions[random.randint(0, len(list_of_best_actions) - 1)]
            if return_q:
                return (best_action, max_q)
            else:
                return best_action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        # Current Q value for state1, action1
        old_q = self.getQ(state1, action1)

        # Best Q_value we can get from state2
        best_q2 = max([self.getQ(state2, a) for a in self.actions])

        # Check if the state1, action1 pair is in the dictionary, update its value
        if (state1, action1) not in self.q:
            self.q[(state1, action1)] = self.alpha * (reward + self.gamma * best_q2)
        else:
            self.q[(state1, action1)] += self.alpha * (reward + self.gamma * best_q2 - old_q)
