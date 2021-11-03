import gym
import neat
import os
import visualize
import numpy as np

# Алгоритм сам определяет сколько входов и сколько выходов требуется
# пока работает только с дискретными входами спортзала

"""
gen - Номер енерации
env - спортзал 
envs - []]

 - геномы
ge = []
net - сеть 
nets = []
cart - ответ сети
carts = []
 - Маска, показывает какие геномы еще живы
do_it = [] 


"""

GYM_NAME = 'LunarLander-v2'
gen = 0
envs = [] #создаю глобально т.к. если постоянно пересоздавать спортзал, он ломается

def test_winer(env,winner,config,protc=50,testov=3,render=False, debug=False, print_genome=False,function="avg"):

    ge = winner
    net = neat.nn.FeedForwardNetwork.create(ge, config)


    def del_files(files):
        for f in files:
            if os.path.exists(f):
                print(f)
                os.remove(f)

    if print_genome:
        del_files(['Digraph.gv','Digraph.gv.pdf','Digraph.gv.png','Digraph.gv.svg'])
        visualize.draw_net(config, ge, view=True)


    rez=0
    for i in range(testov):
        score=0
        observation = env.reset()
        observation = normalization(observation)
        action=0
        done = False

        while not done:
            output = net.activate(observation)
            if env.action_space.dtype.name == 'int64':
                action = output.index(max(output)) # Выбираю вход с самым сильным сигналом

            observation, reward, done, info = env.step(action)
            observation = normalization(observation)

            score += reward


            if render:
                env.render()
        if debug:
            print(f'Игра №{i+1} из {testov} Счет = {score}')
        rez+=score
        if function == "min": # Любой тест меньше порога приводит к провалу
            if 100/config.fitness_threshold*score<protc:
                return False
    # env.close()
    if function == "avg":
        if debug:
            print(f'Средний чет = {round(100/config.fitness_threshold*rez/testov,0)}%  Порог прохождения {protc}%')

        return 100/config.fitness_threshold*rez/testov>=protc
    # return False # временно ввел для продления тренировки сети
    elif function == "min":
        return True
    elif function == "max":
        return True

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    global envs
    global max_popitok

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Выясняю у спортзала число входов и выходов, и правлю конфиг до создания популяции
    envs.append(gym.make(GYM_NAME))
    config.genome_config.num_outputs = envs[0].action_space.n # Число выходов сети рано числу входов спортзала
    config.genome_config.output_keys=[i for i in range(config.genome_config.num_outputs)] # Нумерою выходы
    config.genome_config.num_inputs = envs[0].observation_space.shape[0] # Число входов сети равно числу выходов спортзала # лишний вход, это расстояние от тележки до чекпоинта деленное на 3
    config.genome_config.input_keys = [-i for i in range(config.genome_config.num_inputs)]
    config.fitness_threshold = envs[0].spec.reward_threshold # максимальное состижение в данном спортзале

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-28 (118% avg с увеличениме попыток)')
    #p.config.fitness_criterion='max'
    # p.config = config

    #
    # # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # neat.checkpoint

    f_exit = False
    while not f_exit:
        winner = p.run(eval_genomes, 2000)  # Максимальное число генераций

        # show final stats
        print('\nBest genome:\n{!s}'.format(winner))



        # Если 3 игры средний результат за 3 игры привысит 100% от максимума, выходим
        f_exit=test_winer(env=envs[0],winner=winner,config=config, protc=70, testov=10,render=False, debug=True, print_genome=True, function="min")
        if not f_exit:
            max_popitok += 1 # Если winer не смог пройти тест из 10 игр, увеличиваю число попыток для отбора winer
            print(f"======== Увеличил число попыток, теперь оно равняется {max_popitok}")





    test_winer(env=envs[0],winner=winner,config=config, protc=90, testov=100, render=True, debug=True)
    for i in range(len(envs)):
        envs[0].close()
        envs.pop(0)


# ЗАмеряю абсолютные минимумы и максимумы значений
MAX_observation = []
MIN_observation = []
def min_max_observation(observation):
    global MAX_observation, \
           MIN_observation
    if len(MAX_observation)==0:
        MAX_observation = observation.copy()
        MIN_observation = observation.copy()

    for i in range(len(observation)):
        if observation[i]>MAX_observation[i]:
            MAX_observation[i] = observation[i]
        if observation[i]<MIN_observation[i]:
            MIN_observation[i] = observation[i]
# Собранные значения
# MIN = [ -1.0440816   -0.43393913  -5.1510406   -5.7763205  -40.84389  -13.237666     0.           0.        ]
# MAX = [ 1.0485777    40.912003     5.216288     7.933868    34.291817  10.499603     1.           1.       ]


# Нормализирует данные, не сохраняю дисперсию
def normalization(observation):
    # Сбор статистики
    min_max_observation(observation)


    return observation # отключил нормализацию т.к. нормализация в таком виде только путает сеть

    o_min = [ -1.0440816,   -0.43393913,  -5.1510406,   -8.214328,  -63.89828,  -22.963087,     0.,           0.        ]
    o_max = [ 1.0485777,    178.45638,     5.216288,     13.333334,    73.855255,  14.877298,     1.,           1.       ]

    # максимальные значения отклонений от нуля по каждому ответу сети
    max_mm=[]
    for a,b in zip(o_min,o_max):
        max_mm.append(max(abs(a), abs(b))/10)
    try:
        for i in range(observation.size):
            # observation[i] = 2*((observation[i]-o_min)/(o_max-o_min)-0.5) # приводим все к значению от 1 до -1 # решил что сдвиг путает сеть
            observation[i] = (observation[i]/max_mm[i])  # маштабирует значения в пределах от -1 до 1
    except:
        # если происходит деление на ноль, просто пропускаем т.к. все элементы равны 0 и нормализовывать нечего
        pass
    return observation

max_popitok = 1 # сколько раз алгоритм пробует свои силы для получения значения фитнеса
def eval_genomes(genomes, config):
    global gen
    global envs
    global max_popitok

    gen += 1

    # max_popitok = 3
    # popitki_ge = [] # это будет массив попыток массивов геномов этих попыток
    # for i in range(len(genomes)):
    #     popitki_ge.append([])
    popitki_ge = [[0. for _ in range(len(genomes))] for _ in range(max_popitok)] # это будет массив попыток массивов геномов этих попыток


    for popitka in range(max_popitok): # Хочу выявлять лучшего из N попыток
        # envs = []  # среды с агентами
        ge = []  # геномы
        nets = []
        carts = []
        # graph_net = []
        do_it = [] # Маска, показывает какие геномы еще живы


        # создаем среды
        for i in range(len(genomes)):
            do_it.append(True)
            # genome.fitness = 0
            popitki_ge[popitka][i] = 0
            net = neat.nn.FeedForwardNetwork.create(genomes[i][1], config)
            nets.append(net)
            ge.append(genomes[i][1])

        for i in range(len(ge)):
            if len(envs)-1-i<0:
                envs.append(gym.make(GYM_NAME))  # CartPole-v1  LunarLander-v2 LunarLanderContinuous-v2

        if gen == 1:
            print("action space: {0!r}".format(envs[0].action_space))
            print("observation space: {0!r}".format(envs[0].observation_space))


        score = 0
        f_run = True

        for i in range(len(ge)):
            action = 0#[0.0,0.0]
            observation = envs[i].reset()
            observation = normalization(observation)
            carts.append({'observation': observation})

        while f_run and len(ge) > 0:
            f_run = False
            score += 1

            # Запрашиваем дествие от сети
            for n in range(len(ge)):
                if do_it[n]:
                    # запрашиваем дейсвие от сети
                    temp=(carts[n]['observation'])
                    output = nets[n].activate(temp)
                    if envs[n].action_space.dtype.name == 'int64':
                        action = output.index(max(output)) # Выбираю вход с самым сильным сигналом

                    observation, reward, done, info = envs[n].step(action)  # take a random action
                    observation = normalization(observation)
                    carts[n] = {'observation': observation, 'reward': reward, 'done': done, 'info': info}

                    # ge[n].fitness += reward
                    popitki_ge[popitka][n] += reward

            for i in range(len(ge)):
                if carts[i]['done']:
                    do_it[i] = False

            best_ge_fitness=0 # Номер Генома с лучшим результатом в этом поколении
            for i in range(len(ge)):
                # if ge[i].fitness>ge[best_ge_fitness].fitness:
                if popitki_ge[popitka][i]>popitki_ge[popitka][best_ge_fitness]:
                    best_ge_fitness = i

            for i in range(len(do_it)):
                if do_it[i]:
                    f_run = True

        # раскрываю геномам их фитнес (средний) за сделанные попытки
        if popitka+1 == max_popitok:
            for n in range(len(ge)):
             ge[n].fitness =  sum([popitki_ge[i][n] for i in range(max_popitok)])/popitka # средний результат fitnes за все попытки
            if gen % 1 == 0:  # отрисовываем игру каждого 10 генома только в последней попытке
                # Смотрим на игру лучшего генома в этом поколении
                test_winer(env=envs[0],winner=ge[best_ge_fitness],config=config,protc=100,testov=1, render=True,print_genome=True)
            print(f"MIN =",MIN_observation)
            print(f"MAX =",MAX_observation)




        for i in range(len(do_it)):
            ge.pop(0)
            nets.pop(0)
            carts.pop(0)



# for env in envs:
    #     env.close()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    try:
        run(config_path)
    except Exception as e:
        print('!!!!!!!!!!!!!!!!!',str(e))

