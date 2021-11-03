import math

# Аналог класса из Neat
# Класс ядра нейрона
class DefaultNodeGene:
    def __init__(self, key=0, bias=0.06634488263905369,response=1.0,activation="relu",aggregation="sum"):
        self.key = key
        self.bias = bias
        self.response = response
        self.activation = activation
        self.aggregation = aggregation

    def __str__(self):
        return f"key={self.key}, bias={self.bias},response={self.response},activation={self.activation},aggregation={self.aggregation}"

# Аналог класса из Neat
# Класс связей (аксонов)
class DefaultConnectionGene:
    def __init__(self, key=(-1, 1), weight=0.6116639233947186, enabled=True):
        self.key = key # Dict, анало ключа выше по уровню
        self.weight = weight # float
        self.enabled = enabled

    def __str__(self):
        return f"key={self.key}, weight={self.weight}, enabled={self.enabled}"

# Аналог класса из Neat
class DefaultGenome:
    def __init__(self, key=1,fitnes=0.0):
        self.connections=dict()
        self.nodes=dict() #genome.nodes
        self.fitnes = fitnes
        self.key = key

        # значения на входах. Использоваться при вычислениях сети
        self.IN=dict()

    def __str__(self):
        rez = f"Genome key={self.key} fitnes={self.fitnes}"
        if len(self.nodes)>0:
            rez+=f"\n Nodes:"
        for i in self.nodes:
            rez+=f"\n {i}:{self.nodes[i]}"
        if len(self.nodes)>0:
            rez+=f"\n Connections:"
        for i in self.connections:
            rez+=f"\n {i}:{self.connections[i]}"
        return  rez

    def add_nodes(self,key=0, bias=0.06634488263905369, response=1.0, activation="relu", aggregation="sum"):
        self.nodes[key] = DefaultNodeGene(key=key,bias=bias,response=response,activation=activation,aggregation=aggregation)

    def add_connections(self, key=(-1, 0), weight=-0.5340717852511587, enabled=True):
            self.connections[key] = DefaultConnectionGene(key=key, weight=weight, enabled=enabled)

    # Прямой ход вычислений в созданной нейронке
    def front(self,*args):
        # Упрощенная модель вычислений, для сетей без скрытых слоев

        rez_list=[]
        # Задаем входные значения для сети
        IN={}
        for i, arg in enumerate(args):
            IN[-i-1]=arg
            #print(f"{i}:{arg}")
        #print(self.IN)
        for node_key, node_value in self.nodes.items():
            rez=0
            for connection_key, connection_value in self.connections.items():
                if connection_value.enabled:
                    if connection_key[1] == node_key:
                        rez=node_value.aggregation(rez,IN[connection_key[0]]*connection_value.weight)

            #print(f"node_key={node_key} : {rez}")
            rez_list.append(rez)
        return rez_list



def relu(a):
    return a


def sigmoid(a):
    return 1 / (1 + math.exp(-x))

def summ(*args):
    rez=0
    for i in args:
        rez+=i
    return rez


class Player:
    def __init__(self,key=0,reward=0,genome=DefaultGenome()):
        self.key = key
        self.reward = reward
        self.genome = genome

    def __str__(self):
        return f"Player: {self.key}\nreward={self.reward}\n{self.genome}"



if __name__ == '__main__':
    genom = DefaultGenome(key=111)
    genom.add_nodes(key=0, bias=-0.11099254764388825, response=1.0, activation=cube, aggregation=sum)
    genom.add_connections(key=(-4, 0), weight=0.38262795101523117, enabled=True)
    genom.add_connections(key=(-3, 0), weight=-0.8678197435844905, enabled=True)
    genom.add_connections(key=(-2, 0), weight=1.094726483601257, enabled=True)
    genom.add_connections(key=(-1, 0), weight=1.4229838516702031, enabled=True)
    #print(genom)

    print(genom.front(0,1,33,4))
    print(genom.front(0,1,33,4))
    print(genom.front(0,1,33,4))
    print(genom.front(0,1,33,4))

# Key: 294
# Fitness: 227.61873547417008
# Nodes:
# 0 DefaultNodeGene(key=0, bias=0.06634488263905369, response=1.0, activation=relu, aggregation=sum)
# 1 DefaultNodeGene(key=1, bias=0.11505283573359926, response=1.0, activation=sigmoid, aggregation=sum)
# Connections:
# DefaultConnectionGene(key=(-4, 0), weight=-2.7380754265166463, enabled=True)
# DefaultConnectionGene(key=(-4, 1), weight=-0.346290791953719, enabled=True)
# DefaultConnectionGene(key=(-3, 0), weight=0.28383040202328436, enabled=True)
# DefaultConnectionGene(key=(-3, 1), weight=-0.7280953215465406, enabled=True)
# DefaultConnectionGene(key=(-2, 0), weight=-0.7671497138743135, enabled=True)
# DefaultConnectionGene(key=(-2, 1), weight=-2.6475877086013124, enabled=True)
# DefaultConnectionGene(key=(-1, 0), weight=-0.5340717852511587, enabled=True)
# DefaultConnectionGene(key=(-1, 1), weight=0.6116639233947186, enabled=True)
#
#
# {(-1, 0): <neat.genes.DefaultConnectionGene object at 0x000002D2AB4242B0>, (-1, 1): <neat.genes.DefaultConnectionGene object at 0x000002D2AB424310>, (-2, 0): <neat.genes.DefaultConnectionGene object at 0x000002D2AB424370>, (-2, 1): <neat.genes.DefaultConnectionGene object at 0x000002D2AB4243D0>, (-3, 0): <neat.genes.DefaultConnectionGene object at 0x000002D2AB424430>, (-3, 1): <neat.genes.DefaultConnectionGene object at 0x000002D2AB424490>, (-4, 0): <neat.genes.DefaultConnectionGene object at 0x000002D2AB4244F0>, (-4, 1): <neat.genes.DefaultConnectionGene object at 0x000002D2AB424550>}