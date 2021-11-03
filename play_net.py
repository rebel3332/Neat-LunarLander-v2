import gym
from NETT import *

player = Player()
player.genome.add_nodes(key=0, bias=-0.018745652515057276, response=1.0, activation=sigmoid, aggregation=summ)
player.genome.add_connections(key=(-4, 0), weight=0.8149045426962688, enabled=True)
player.genome.add_connections(key=(-3, 0), weight=1.929706747197698, enabled=True)
# player.genome.add_connections(key=(-1, 0), weight=0.04467591594566367, enabled=True)
# player.genome.add_connections(key=(-1, 0), weight=3.4570106630280844, enabled=False)
# player.genome.add_nodes(key=0, bias=-0.37989151195126525, response=1.0, activation=sigmoid, aggregation=summ)
# player.genome.add_connections(key=(-4, 0), weight=3.339475032706657, enabled=True)
# player.genome.add_connections(key=(-3, 0), weight=1.6032325738049862, enabled=True)
# player.genome.add_connections(key=(-2, 0), weight=2.683521964436893, enabled=True)




env = gym.make('CartPole-v1')
while True:
    player.reward=0
    env.reset()
    action=0
    observation, reward, done, info = env.step(action)
    while not done:
        action=player.genome.front(*observation)
        action=1 if action[0]>0.5 else 0
        observation, reward, done, info = env.step(action)
        # print(observation)
        player.reward+=reward
        env.render()
    print('Счет = ',player.reward)

