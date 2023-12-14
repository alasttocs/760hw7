import numpy as np

# Define the MDP parameters
states = {'A': 0, 'B': 1}
actions = ['Move', 'Stay']
gamma = 0.8
alpha = 0.5
steps = 200
epsilon = 0.5

Q = np.zeros((len(states), len(actions)))
cur_state = states['A']

for _ in range(steps):
    cur_action = np.argmax(Q[cur_state])
    if cur_action == 0:
        if cur_state == states['A']:
            next_state = states['B']
        else:
            next_state = states['A']
        reward = 0
    else:
        next_state = cur_state
        reward = 1

    Q[cur_state][cur_action] = (1 - alpha) * Q[cur_state][cur_action] + \
        alpha * (reward + gamma * np.max(Q[next_state]))
    cur_state = next_state
print(Q)


Q = np.zeros((len(states), len(actions)))
cur_state = states['A']

for _ in range(steps):
    cur_action = np.argmax(Q[cur_state])

    if np.random.rand() < epsilon:
        cur_action = np.random.choice([0, 1])
    else:
        cur_action = np.argmax(Q[cur_state])
    if cur_action == 0:
        if cur_state == states['A']:
            next_state = states['B']
        else:
            next_state = states['A']
        reward = 0
    else:
        next_state = cur_state
        reward = 1

    Q[cur_state][cur_action] = (1 - alpha) * Q[cur_state][cur_action] + \
        alpha * (reward + gamma * np.max(Q[next_state]))
    cur_state = next_state
print(Q)
