from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random_player1(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
    

    def play_against_random_player2(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.random_step()
        if done:
            if status == self.STATUS_WIN:
                status = self.STATUS_LOSE
            elif status == self.STATUS_TIE:
                status = self.STATUS_TIE
            else:
                raise ValueError("???")  
            
        if not done and self.turn == 2:
            state, s2, done = self.step(action)
            status = s2          

        return state, status, done
    
    #==================Part 2 Play================================#
    
    def play_against_train_player1(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            actionOther, logprob = select_action(policy_trained, state)
            state, s2, done = self.step(actionOther)
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
    

    def play_against_train_player2(self, action):
        """Play a move, and then have a random agent play the next move."""
        actionOther, logprob = select_action(policy_trained, state)
        state, status, done = self.step(actionOther)   
        if done:
            if status == self.STATUS_WIN:
                status = self.STATUS_LOSE
            elif status == self.STATUS_TIE:
                status = self.STATUS_TIE
            else:
                raise ValueError("???")  
            
        if not done and self.turn == 2:
            state, s2, done = self.step(action)
            status = s2          

        return state, status, done


class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )        

    def forward(self, x):
        x = self.layer(x)
        x = F.softmax(x, dim=-1)
        return x


def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    size = len(rewards)
    result = [0] * size
    i = size - 1
    while i >= 0:
        if i == size - 1:
            result[i] = float(rewards[i])
        else:
            result[i] = rewards[i] + gamma * result[i+1]
        i -= 1
    return result

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 1, 
            Environment.STATUS_INVALID_MOVE: -50,
            Environment.STATUS_WIN         : 50,
            Environment.STATUS_TIE         : -5,
            Environment.STATUS_LOSE        : -10
    }[status]

def train(policy, env, gamma=0.85, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    number_of_episode = []
    average_return = []
    
    win_rate = []
    lose_rate = []
    tie_rate = []

    win_rate2 = []
    lose_rate2 = []
    tie_rate2 = []    

    for i_episode in count(1):
        
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            
            if i_episode % 2 == 1: #train first half of the iteration as player 1
                state, status, done = env.play_against_random_player1(action)
            else: #second half of the iteration as player 2
                state, status, done = env.play_against_random_player2(action)
                
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)
         
        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            number_of_episode.append(i_episode)

            average_return.append(running_reward/log_interval)
            win, lose, tie, invalid_move = games_play_against_random_player1(policy, env)
            win2, lose2, tie2, invalid_move2 = games_play_against_random_player2(policy, env)
            # for player 1
            win_rate.append(win/100.0)
            lose_rate.append(lose/100.0)
            tie_rate.append(tie/100.0)
            
            # for player 2
            win_rate2.append(win2/100.0)
            lose_rate2.append(lose2/100.0)
            tie_rate2.append(tie2/100.0)        
            
            
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            running_reward = 0
        
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt_part2/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        
        if i_episode == 50000:
            #part 5: plot average return
            plt.figure()
            plt.plot(number_of_episode, average_return)
            plt.xlabel("number of episode")
            plt.ylabel("average return")
            plt.title("Learning curve of Tic-Tac-Toe model")
            plt.savefig("training_curve.png")
            
            # for player 1
            plt.figure()
            plt.plot(number_of_episode, win_rate , label = "win rate")
            plt.plot(number_of_episode, lose_rate, label = "loss rate")
            plt.plot(number_of_episode, tie_rate, label = "tie rate")
            plt.xlabel("number of episode")
            plt.ylabel("win/loss/tie rates")
            plt.title("win/loss/tie rates over episode for player1")
            plt.legend()
            plt.savefig("player1.png")
            
            
            # for player 2
            plt.figure()
            plt.plot(number_of_episode, win_rate2 , label = "win rate")
            plt.plot(number_of_episode, lose_rate2, label = "loss rate")
            plt.plot(number_of_episode, tie_rate2, label = "tie rate")
            plt.xlabel("number of episode")
            plt.ylabel("win/loss/tie rates")
            plt.title("win/loss/tie rates over episode for player2")
            plt.legend()
            plt.savefig("player2.png")            
            
            return
    

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def games_play_against_random_player2(policy, env):
    """
    Use learned policy to play 100 games against random. count the number of times that 
    the agent win/lose/tie. 
    """
    win = 0
    lose = 0
    tie = 0
    invalid_move = 0
    
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random_player2(action)
            if status == 'inv':
                invalid_move += 1
            
        if status == 'win':
            win += 1
        elif status == 'lose':
            lose += 1
        else:
            tie += 1
            
    return win, lose, tie, invalid_move

def games_play_against_random_player1(policy, env):
    """
    Use learned policy to play 100 games against random. count the number of times that 
    the agent win/lose/tie. 
    """
    win = 0
    lose = 0
    tie = 0
    invalid_move = 0
    
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random_player1(action)
            if status == 'inv':
                invalid_move += 1
            
        if status == 'win':
            win += 1
        elif status == 'lose':
            lose += 1
        else:
            tie += 1
            
    return win, lose, tie, invalid_move

def display_games(policy, env, player):
    """
    Display five games that the trained agent plays against the random policy.
    player = 1 if it's player1
    player = 2 if it's player2
    """
    for i in range(5):
        print("Game" + str(i + 1) + ":")
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            if player == 1:
                state, status, done = env.play_against_random_player1(action)
            if player == 2:
                state, status, done = env.play_against_random_player2(action)                
            env.render()
        print("******************") 

#==============================================================================#
#                      Part 2 Play Self                                        #
#==============================================================================#

def train_selfPlay(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    number_of_episode = []
    average_return = []
    
    win_rate = []
    lose_rate = []
    tie_rate = []

    win_rate2 = []
    lose_rate2 = []
    tie_rate2 = []    

    for i_episode in count(1):
        
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            
            state, status, done = env.play_against_train_player1(action)
                
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)
         
        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            number_of_episode.append(i_episode)

            average_return.append(running_reward/log_interval)
            win, lose, tie, invalid_move = games_play_against_random_player1(policy, env)
            win2, lose2, tie2, invalid_move2 = games_play_against_random_player2(policy, env)
            # for player 1
            win_rate.append(win/100.0)
            lose_rate.append(lose/100.0)
            tie_rate.append(tie/100.0)
            
            # for player 2
            win_rate2.append(win2/100.0)
            lose_rate2.append(lose2/100.0)
            tie_rate2.append(tie2/100.0)        
            
            
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            running_reward = 0
        
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt_part2/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        
        if i_episode == 50000:
            #part 5: plot average return
            plt.figure()
            plt.plot(number_of_episode, average_return)
            plt.xlabel("number of episode")
            plt.ylabel("average return")
            plt.title("Learning curve of Tic-Tac-Toe model")
            plt.savefig("p2_training_curve.png")
            
            # for player 1
            plt.figure()
            plt.plot(number_of_episode, win_rate , label = "win rate")
            plt.plot(number_of_episode, lose_rate, label = "loss rate")
            plt.plot(number_of_episode, tie_rate, label = "tie rate")
            plt.xlabel("number of episode")
            plt.ylabel("win/loss/tie rates")
            plt.title("win/loss/tie rates over episode for player1")
            plt.legend()
            plt.savefig("p2_player1.png")
            
            
            # for player 2
            plt.figure()
            plt.plot(number_of_episode, win_rate2 , label = "win rate")
            plt.plot(number_of_episode, lose_rate2, label = "loss rate")
            plt.plot(number_of_episode, tie_rate2, label = "tie rate")
            plt.xlabel("number of episode")
            plt.ylabel("win/loss/tie rates")
            plt.title("win/loss/tie rates over episode for player2")
            plt.legend()
            plt.savefig("p2_player2.png")            
            
            return
    
if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()

    if len(sys.argv) == 1:
        #train with self-play
        policy_trained = Policy()
        weights = torch.load("ttt/policy-%d.pkl" % 50000)
        policy_trained.load_state_dict(weights)        
        train_selfPlay(policy, env)
        ## `python tictactoe.py` to train the agent
        #train(policy, env)
    else:
        ## `python tictactoe.py <ep>` to print the first move distribution
        ## using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))
