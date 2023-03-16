# main algorithm
import random as rand
import numpy as np
from functools import cache, lru_cache
import asyncio  # managing coroutines
import networkx as nx  # generating graphs
import matplotlib.pyplot as plt  # visualizing graphs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import PIL  # image processing
from hyperparams import *


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: float, y: float) -> float:
    return 1 / (1 + np.exp(-x+y))


def null(x) -> float:
    return 0.0

def llun(x) -> float:
    return 1.0

def dist_n(entity) -> float:
    return entity.y/CANVAS_DIM

def dist_s(entity) -> float:
    return 1.0 - entity.y/CANVAS_DIM

def dist_e(entity) -> float:
    return 1.0 - entity.x/CANVAS_DIM

def dist_w(entity) -> float:
    return entity.x/CANVAS_DIM

def dist_fwd(entity) -> float:
    dist_dict = [dist_n, dist_e, dist_s, dist_w]
    return dist_dict[entity.direction](entity)

def closest_n(entity) -> float:
    x, y = entity.x, entity.y
    nearest_y= max([e.y for e in entities if ((e != entity) and (e.x == x) and (e.y <= y))] + [-float('inf')])
    return sigmoid_prime(y - nearest_y, CANVAS_DIM/8)

def closest_s(entity) -> float:
    x, y = entity.x, entity.y
    nearest_y= min([e.y for e in entities if ((e != entity) and (e.x == x) and (e.y >= y))] + [float('inf')])
    return sigmoid_prime(nearest_y - y, CANVAS_DIM/8)

def closest_e(entity) -> float:
    x, y = entity.x, entity.y
    nearest_x= min([e.y for e in entities if ((e != entity) and (e.y == y) and (e.x >= x))] + [float('inf')])
    return sigmoid_prime(nearest_x - x, CANVAS_DIM/8)

def closest_w(entity) -> float:
    x, y = entity.x, entity.y
    nearest_x= max([e.y for e in entities if ((e != entity) and (e.y == y) and (e.x <= x))] + [-float('inf')])
    return sigmoid_prime(x - nearest_x, CANVAS_DIM/8)

def closest_fwd(entity) -> float:
    closest_dict = [closest_n, closest_e, closest_s, closest_w]
    return closest_dict[entity.direction](entity)

def population(entity) -> float:
    return (entities.alive / INITIAL_GEN_POP)


def move_n(neuron):
    if neuron.value >= 0.5:
        neuron.master.entity.y -= neuron.value * K_VELOCITY

def move_s(neuron):
    if neuron.value >= 0.5:
        neuron.master.entity.y += neuron.value * K_VELOCITY

def move_e(neuron):
    if neuron.value >= 0.5:
        neuron.master.entity.x += neuron.value * K_VELOCITY

def move_w(neuron):
    if neuron.value >= 0.5:
        neuron.master.entity.x -= neuron.value * K_VELOCITY

def move_fwd(neuron):
    if neuron.value >= 0.5:
        move_list = [move_n, move_e, move_s, move_w]
        move_list[neuron.master.entity.direction](neuron)

def rotate(neuron):
    if neuron.value >= 0.25:
        neuron.master.entity.direction = (neuron.master.entity.direction  - 1) % 4
        return
    if neuron.value <= 0.75:
        neuron.master.entity.direction = (neuron.master.entity.direction  + 1) % 4
        return


class Neuron:
    def __init__(self, master, identifier: tuple[int, int], bias: float, name=""):  # master: 종속된 Entity
        self.master = master
        self.id = identifier
        self.value = bias
        self.bias = bias
        self.name = name

    def copy(self, master):
        return Neuron(master, self.id, self.bias, self.name)

    def reset(self):
        self.value = self.bias


class InputNeuron(Neuron):
    def __init__(self, master, identifier: tuple[int, int], bias: float, inputf=null, name=""):
        Neuron.__init__(self, master, identifier, bias, name)
        self.input = inputf

    def get_input(self):
        self.value = self.input(self.master.entity)

    def copy(self, master):
        return InputNeuron(master, self.id, self.bias, self.input, self.name)


class OutputNeuron(Neuron):
    def __init__(self, master, identifier: tuple[int, int], bias: float, function=None, name=""):
        Neuron.__init__(self, master, identifier, bias, name)
        self.function = function

    def copy(self, master):
        return OutputNeuron(master, self.id, self.bias, self.function, self.name)

    def act(self):
        self.function(self)
        self.reset()


InputNeurons = [
    InputNeuron(None, (0,0), 0.0, null, "0"),
    InputNeuron(None, (0,1), 0.0, llun, "1"),
    InputNeuron(None, (0,2), 0.0, dist_n, "dist_n"),
    InputNeuron(None, (0,3), 0.0, dist_s, "dist_s"),
    InputNeuron(None, (0,4), 0.0, dist_e, "dist_e"),
    InputNeuron(None, (0,5), 0.0, dist_w, "dist_w"),
    InputNeuron(None, (0,6), 0.0, dist_fwd, "dist_fwd"),
    InputNeuron(None, (0,7), 0.0, closest_n, "closest_n"),
    InputNeuron(None, (0,8), 0.0, closest_s, "closest_s"),
    InputNeuron(None, (0,9), 0.0, closest_e, "closest_e"),
    InputNeuron(None, (0,10), 0.0, closest_w, "closest_w"),
    InputNeuron(None, (0,11), 0.0, closest_fwd, "closest_fwd"),
    InputNeuron(None, (0,12), 0.0, null, "nearest_dx"),
    InputNeuron(None, (0,13), 0.0, null, "nearest_dy"),
    InputNeuron(None, (0,14), 0.0, population, "population"),
    InputNeuron(None, (0,15), 0.0, null, "oscillator")            
]  # 미리 지정한 InputNeuron 들의 list
OutputNeurons = [
    OutputNeuron(None, (2,0), 0.0, null, "null"), 
    OutputNeuron(None, (2,1), 0.0, move_n, "move_n"), 
    OutputNeuron(None, (2,2), 0.0, move_s, "move_s"), 
    OutputNeuron(None, (2,3), 0.0, move_e, "move_e"), 
    OutputNeuron(None, (2,4), 0.0, move_w, "move_w"), 
    OutputNeuron(None, (2,5), 0.0, move_fwd, "move_fwd"), 
    OutputNeuron(None, (2,6), 0.0, rotate, "rotate"), 
]
''' OutputNeuron(None, (2,7), 0.0, null), 
    OutputNeuron(None, (2,8), 0.0, null), 
    OutputNeuron(None, (2,9), 0.0, null), 
    OutputNeuron(None, (2,10), 0.0, null), 
    OutputNeuron(None, (2,11), 0.0, null), 
    OutputNeuron(None, (2,12), 0.0, null), 
    OutputNeuron(None, (2,13), 0.0, null), 
    OutputNeuron(None, (2,14), 0.0, null), 
    OutputNeuron(None, (2,15), 0.0, null), '''
#] # 미리 지정한 OutputNeuron 들의 list

n_InputN = len(InputNeurons)
n_OutputN = len(OutputNeurons)


def interp_gene(network): 
    # set up the network using the gene
    l_c = [[], [], []]
    d_ip = {}
    d_op = {}
    for n in range(0, BRAIN_SIZE):
        g = network.gene[6 * n: 6 * n + 6]
        g_t = int(g[0], 16) % 4
        if g_t == 3:
            continue
        g_so = int(g[1], 16)
        g_si = int(g[2], 16)
        g_w = MAX_WEIGHT * (float(int(g[3:-1], 16)) - 127.5) / 127.5
        if g_t == 0:
            g_so = g_so % n_InputN
            g_si = g_si % n_OutputN
            d_ip[g_so] = 0
            d_op[g_si] = 0
            l_c[0].append((g_so, g_si, g_w))
        elif g_t == 1:
            g_so = g_so % n_InputN
            g_si = g_si % N_INNER
            d_ip[g_so] = 0
            l_c[1].append((g_so, g_si, g_w))
        else:
            g_so = g_so % N_INNER
            g_si = g_si % n_OutputN
            d_op[g_si] = 0
            l_c[2].append((g_so, g_si, g_w))
    for ip in d_ip:
        network.input_neurons.append(InputNeurons[ip].copy(network))
    for op in d_op:
        network.output_neurons.append(OutputNeurons[op].copy(network))
    for conn in l_c[0]:
        for conn in l_c[0]:
            network.connections[0].append((network.input_neurons[list(d_ip.keys()).index(conn[0])], network.output_neurons[list(d_op.keys()).index(conn[1])], conn[2]))
    for conn in l_c[1]:
        for conn in l_c[1]:
            network.connections[1].append((network.input_neurons[list(d_ip.keys()).index(conn[0])], network.inner_neurons[conn[1]], conn[2]))
    for conn in l_c[2]:
        for conn in l_c[2]:
            network.connections[2].append((network.inner_neurons[conn[0]], network.output_neurons[list(d_op.keys()).index(conn[1])], conn[2]))


class Network:
    def __init__(self, entity, n_inner: int, gene: str):
        self.entity = entity
        self.gene = gene
        self.input_neurons = []
        self.output_neurons = []
        self.inner_neurons = []
        for i in range(n_inner):
            self.inner_neurons.append(Neuron(self, (1, i), 0.0, f"innner{i}"))
        self.connections = [[], [],
                            []]  # connections[0]: input->output, connections[1]: input->inner, connections[2]: inner->output
        # connection = (source, sink, weight)
        interp_gene(self)

    def run(self):
        # calculate neurons
        for inp in self.input_neurons:
            inp.get_input()
        for i in (0, 1, 2):
            conns = self.connections[i]
            sincs = {}
            for conn in conns:
                conn[1].value += conn[0].value * conn[2]
                sincs[conn[1]] = 0
            if i == 0:
                continue
            if i == 2:
                sincs = self.output_neurons
            for sinc in sincs:
                sinc.value = sigmoid(sinc.value)
        # run output
        for out in self.output_neurons:
            out.act()
        # reset neurons
        for inn in self.inner_neurons:
            inn.reset()

    def makeGraph(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from([str(neu.id) for neu in self.input_neurons+self.inner_neurons+self.output_neurons])
        G.add_edges_from([(str(conn[0].id), str(conn[1].id)) for conn in self.connections[0]+self.connections[1]+self.connections[2]])
        layout = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, layout, nodelist=[str(inp.id) for inp in self.input_neurons], node_color="tab:red")
        nx.draw_networkx_nodes(G, layout, nodelist=[str(inn.id) for inn in self.inner_neurons], node_color="tab:gray")
        nx.draw_networkx_nodes(G, layout, nodelist=[str(out.id) for out in self.output_neurons], node_color="tab:blue")

        nx.draw_networkx_edges(G, layout, edgelist=[(str(conn[0].id), str(conn[1].id)) for conn in self.connections[0]+self.connections[1]+self.connections[2]],  arrows=True, arrowstyle="->", edge_color="tab:gray")
        nx.draw_networkx_labels(G, layout, {str(inp.id): inp.name for inp in self.input_neurons}, font_size=16)
        nx.draw_networkx_labels(G, layout, {str(inn.id): inn.name for inn in self.inner_neurons}, font_size=16)
        nx.draw_networkx_labels(G, layout, {str(out.id): out.name for out in self.output_neurons}, font_size=16)
        nx.draw_networkx_edge_labels(G, layout, {(str(conn[0].id), str(conn[1].id)): np.round(conn[2], 3) for conn in self.connections[0]+self.connections[1]+self.connections[2]}, font_size=9)

        plt.tight_layout()
        plt.axis("off")
        plt.show()

    def printGene(self):
        print(self.gene)


def mutateGene(gene):
    # apply mutation with prob. P_MUTATION for every bit in gene
    new_gene = ''
    for char in gene:
        if rand.random() <= P_MUTATION:
             new_gene += format(rand.randint(0, 15), 'x')
        else:
             new_gene += char
    return new_gene

def randGene():
    # generate random gene
    new_gene = ''
    for i in range(6*BRAIN_SIZE):
        new_gene += format(rand.randint(0, 15), 'x')
    return new_gene


class Entity:
    def __init__(self, x, y):
        self.network = None
        self.x = x
        self.y = y
        self.direction = 0
        self.alive = True
        self.point = 0

    def reset(self):  
        # reset Entity for new round
        self.network = None
        self.x = 0
        self.y = 0
        if not self.alive:
            self.alive = True
            entities.alive += 1
        self.point = 0

    def status_check(self):
        # check if Entity is alive
        if self.alive:
            if self.point >= 0:
                self.alive = True
                ##
            else:
                self.alive = False
                entities.alive -= 1
        else:
            pass

    def run(self):
        if self.alive:
            self.network.run()
        self.status_check()


if __name__ == "__main__":
    net = Network(None, N_INNER, '01a000029fff1029b82b394813579b')
    net2 = Network(None, N_INNER, '011fff300000300000300000300000')
    # net.run()
    net.makeGraph()
    print('done')
    # 테스트할 거 있으면