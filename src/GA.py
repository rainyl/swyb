import random
from itertools import accumulate
import numpy as np
from tqdm import tqdm


class GA(object):
    def __init__(self, pop_size, chrom_length):
        self.pop_size = pop_size
        self.chrom_length = chrom_length
        self.generation = 500
        self.pops = np.zeros((pop_size, chrom_length))
        self.std = 0.5
        self.p_cross = 0.6
        self.p_mutation = 0.01
        self.ranges = [[100, 130],  # WM
                       [0, 50],  # WUM
                       [0, 50],  # WLM
                       [0.5, 1.5],  # KC
                       [0, 1],  # C
                       [1, 3],  # B
                       [0.01, 1],  # FE  初始含水系数
                       [0, 100],  # SM
                       [1, 3],  # EX
                       [0, 0.2],  # KG
                       [0.5, 1.5],  # KKG
                       [0, 0.1],  # KSS
                       [0, 1.5]]  # KKSS

    def init_pop(self):
        for i, r in enumerate(self.ranges):
            self.pops[:, i] = np.array([random.uniform(r[0], r[1]) for j in range(self.pop_size)])
        return self.pops

    def evaluate(self, values):
        fit_values = np.sin(self.pops)
        return np.where(fit_values > self.std)

    def selection(self, weight):
        weight = list(weight)
        weight.sort()
        acc = list(accumulate(weight))
        rand = np.array([
            random.random(0, 1) for i in range(self.pop_size)
        ])
        rand.sort()
        new_pop = []
        for r in rand:
            for j in range(1, len(acc)+1):
                if acc[j-1] < r < acc[j]:
                    new_pop.append(r)
        return new_pop

    def crossover(self, points=1):
        locs = [round(random.random(0, 1), 1) for i in range(points)]
        locs = [int(i*self.chrom_length) for i in locs]
        loc = int(self.chrom_length/2)
        random.shuffle(self.pops)  # 打乱种群染色体
        # 单点交叉
        for i in range(int(self.pop_size*self.p_cross)):
            tmp = self.pops[i][loc:]
            self.pops[i+1][loc:] = tmp
            self.pops[i][loc:] = self.pops[i+1][loc:]
        return self.pops

    def mutation(self):
        counter = []  # 记录变异位置
        for i in range(self.pop_size):
            for j in self.chrom_length:
                if random.random() < self.p_mutation:
                    if self.pops[i][j] == 0:
                        self.pops[i][j] = 1
                    else:
                        self.pops[i][j] = 0
                    counter.append([i, j])
        return counter

    def go(self):
        self.pops = self.init_pop()
        for i in tqdm(range(self.generation)):
            val = self.evaluate()
            self.pops = self.selection(val)
            self.crossover()
            self.mutation()

    @classmethod
    def float2bin(cls, _float, _len=2):
        _int = int(_float)
        _float = round(_float, _len) - _int
        # res = str(bin(_int))[2:] + '.'
        res = str(bin(_int))[2:]
        for i in range(_len):
            _float = round(_float * 2, _len)
            res += str(int(_float))
            _float -= int(_float)
        return res

    @classmethod
    def bin2float(cls, _float, _len=2):
        _float = str(_float)
        exp = ''
        for i in range(-_len, len(_float)-_len):
            exp += '+{}*2**{}'.format(_float[i], i)
        res = eval(exp)
        return res


if __name__ == '__main__':
    ga = GA(200, 13)
    ga.go()
    x = 22.8125
    print(GA.float2bin(x, 4))
    print(GA.bin2float(GA.float2bin(x, 4), 4))
    print("<DEBUG>")
