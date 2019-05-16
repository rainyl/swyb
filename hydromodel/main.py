import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from itertools import accumulate
from tqdm import tqdm
import sqlite3


class Xinanjiang(object):
    def __init__(self, _args, _data):
        self.p, self.E, self.q = np.array(_data['p']), \
                                 np.array(_data['e']), \
                                 np.array(_data['q'])  # 降雨 蒸发 径流
        self.days = _data['days']
        self.__len = len(self.p)
        (self.WM, self.WUM, self.WLM, self.KC, self.C, self.B, self.IMP, self.FE) = \
            (_args['WM'], _args['WUM'], _args['WLM'],
             _args['KC'], _args['C'], _args['B'], _args['IMP'], _args['FE'])
        self.WDM = self.WM - self.WUM - self.WLM

        # for evaporation
        self.wu, self.wl, self.wd, self.w, self.eu, self.el, self.ed, self.e, self.ep = \
            np.zeros((9, self.__len + 1))
        self.pe = np.zeros((1, self.__len + 1)).flatten()
        self.wu[0], self.wl[0], self.wd[0] = self.FE * np.array([self.WUM, self.WLM, self.WDM])
        self.w[0] = np.sum([self.wu[0], self.wl[0], self.wd[0]])
        self.ep = self.KC * np.array(self.E)
        # self.p = (1-self.IMP)*self.p  # 考虑不透水区域，其所占比例为IMP

        # for runoff
        self.W = self.WM * self.FE
        self.WMM = self.WM * (1 + self.B) / (1 - self.IMP)
        self.r = np.zeros((1, self.__len + 1)).flatten()

        # 水源划分参数定义
        self.SM, self.EX, self.KG, self.KKG, self.KSS, self.KKSS = \
            _args['SM'], _args['EX'], _args['KG'], _args['KKG'], _args['KSS'], _args['KKSS']
        self.FR, self.S, self.R, self.RS, self.RI, self.RG, self.Q = np.zeros((7, self.__len + 1))
        self.S[0] = self.SM * self.FE

        # 汇流参数定义
        self.UH = _args['UH']
        self.A = _args['A']
        self.delta_t = 24
        self.QRS, self.QRI, self.QRG, self.QRT, self.Q = np.zeros((5, self.__len + 1))
        self.QRI[0], self.QRG[0] = 0.5, 0.0

    def refresh_args(self, _args):
        self.__len = len(self.p)
        (self.WM, self.WUM, self.WLM, self.KC, self.C, self.B, self.IMP, self.FE) =\
            (_args['WM'], _args['WUM'], _args['WLM'],
             _args['KC'], _args['C'], _args['B'], _args['IMP'], _args['FE'])
        self.WDM = self.WM - self.WUM - self.WLM

        # for evaporation
        self.wu, self.wl, self.wd, self.w, self.eu, self.el, self.ed, self.e, self.ep = \
            np.zeros((9, self.__len+1))
        self.pe = np.zeros((1, self.__len+1)).flatten()
        self.wu[0], self.wl[0], self.wd[0] = self.FE * np.array([self.WUM, self.WLM, self.WDM])
        self.w[0] = np.sum([self.wu[0], self.wl[0], self.wd[0]])
        self.ep = self.KC * np.array(self.E)
        # self.p = (1-self.IMP)*self.p  # 考虑不透水区域，其所占比例为IMP

        # for runoff
        self.W = self.WM * self.FE
        self.WMM = self.WM * (1 + self.B)/(1 - self.IMP)
        self.r = np.zeros((1, self.__len+1)).flatten()

        # 水源划分参数定义
        self.SM, self.EX, self.KG, self.KKG, self.KSS, self.KKSS =\
            _args['SM'], _args['EX'], _args['KG'], _args['KKG'], _args['KSS'], _args['KKSS']
        self.FR, self.S, self.R, self.RS, self.RI, self.RG, self.Q = np.zeros((7, self.__len+1))
        self.S[0] = self.SM * self.FE

        # 汇流参数定义
        self.UH = _args['UH']
        self.A = _args['A']
        self.delta_t = 24
        self.QRS, self.QRI, self.QRG, self.QRT, self.Q = np.zeros((5, self.__len+1))
        self.QRI[0], self.QRG[0] = 0.5, 0.0

    def runoff(self):
        # 蒸发计算
        # print('<debug> ###蒸发计算开始###\n')
        for i in range(self.__len):
            wu, p, ep, wl, wd = self.wu[i], self.p[i], self.ep[i], self.wl[i], self.wd[i]
            p = p * (1 - self.IMP)  # 考虑不透水区域，其所占比例为IMP
            if wu + p >= ep:
                self.eu[i] = ep
                self.el[i] = 0
                self.ed[i] = 0
            else:
                eu = wu + p
                self.eu[i] = eu
                if wl >= self.C * self.WLM:
                    self.el[i] = (ep - eu) * wl / self.WLM
                    self.ed[i] = 0
                else:
                    if wl >= self.C * (ep - eu):
                        self.el[i] = self.C * (ep - eu)
                        self.ed[i] = 0
                    else:
                        el = wl
                        self.el[i] = el
                        self.ed[i] = self.C * (ep - eu) - wl
                        if self.ed[i] > wd:
                            self.ed[i] = wd
            self.e[i] = np.sum([self.eu[i], self.el[i], self.ed[i]])
            # print('<debug> ###蒸发计算结束###\n')

            # 产流计算
            a = self.WMM * (1 - (1 - self.w[i] / self.WM) ** (1 / (1 + self.B)))
            pe = p * (1 - self.IMP) - self.e[i]
            self.pe[i] = pe
            if pe < 0:
                self.r[i] = 0
            else:
                if a + pe <= self.WMM:
                    self.r[i] = pe + self.w[i] - self.WM + self.WM * (
                                1 - (pe + a) / self.WMM) ** (self.B + 1)
                else:
                    self.r[i] = pe - (self.WM - self.w[i])

            # 含水量计算
            wu, wl, wd, p = self.wu[i], self.wl[i], self.wd[i], self.p[i]
            eu, el, ed, r = self.eu[i], self.el[i], self.ed[i], self.r[i]
            tmp = wu + p - eu - r
            if tmp < self.WUM:
                self.wu[i+1] = tmp
                self.wl[i+1] = wl - el
                self.wd[i+1] = wd - ed
                self.w[i+1] = np.sum([self.wu[i+1], self.wl[i+1], self.wd[i+1]])
                continue
            self.wu[i+1] = self.WUM
            tmp = self.wu[i + 1] + p - eu - r - self.WUM
            if wl - el + tmp < self.WLM:
                self.wl[i+1] = wl - el + tmp
                self.wd[i+1] = wd - ed
                self.w[i+1] = np.sum([self.wu[i+1], self.wl[i+1], self.wd[i+1]])
                continue
            self.wl[i+1] = self.WLM
            tmp = self.wl[i+1] - el + tmp - self.WLM
            if wd - ed + tmp < self.WDM:
                self.wd[i+1] = tmp
                self.w[i+1] = np.sum([self.wu[i+1], self.wl[i+1], self.wd[i+1]])
                continue
            self.wd[i+1] = self.WDM
            self.w[i+1] = np.sum([self.wu[i+1], self.wl[i+1], self.wd[i+1]])

    def water_source_split(self):
        # print(("|')<debug> ###水源划分计算开始###\n')
        for i in range(self.__len):
            # s0 = self.data.loc[0, 'S']
            MS = self.SM * (1 + self.EX)
            pe, w, s, r = self.pe[i], self.w[i], self.S[i], self.r[i]
            # s0, fr0 = self.data.loc[i-1, ['S', 'FR']]
            if pe <= 0:
                fr = 1 - (1 - w / self.WM) ** (self.B / (1 + self.B))
                self.FR[i] = fr
                self.RS[i] = 0
                self.RI[i] = s * self.KSS * fr
                self.RG[i] = s * self.KG * fr
                self.S[i+1] = (1 - self.KSS - self.KG) * s
            else:
                fr = r / pe
                self.FR[i] = fr
                # AU = MS * (1 - (1 - (s0*(fr0/fr)) / self.SM)**(1 / (1 + self.EX)))
                AU = MS * (1 - (1 - s / self.SM) ** (1 / (1 + self.EX)))
                if pe + AU < MS:
                    # self.data.loc[i, 'RS'] = fr * (pe + s0 * fr0 / fr - self.SM + self.SM * (1 - (pe / AU) / MS) ** (
                    #             self.EX + 1))
                    tmp = self.SM * (1 - (pe + AU) / MS) ** (self.EX + 1)
                    self.RS[i] = fr * (pe + s - self.SM + tmp)
                    self.RI[i] = fr * self.KSS * (self.SM - tmp)
                    self.RG[i] = fr * self.KG * (self.SM - tmp)
                    self.S[i+1] = (1 - self.KSS - self.KG) * (self.SM - tmp)
                else:
                    self.RS[i] = fr * (pe - self.SM + s)
                    self.RI[i] = fr * self.SM * self.KSS
                    self.RG[i] = fr * self.KG * self.SM
                    self.S[i+1] = self.SM * (1 - self.KSS * self.KG)
        # print(("|')<debug> ###水源划分计算结束###\n')

    def concentrate(self):
        # 汇流计算
        # print(("|')<debug> ###汇流计算开始###\n')
        for i in range(self.__len):
            for j, u in enumerate(self.UH):
                rs = self.RS[i]
                r_imp = self.p[i] * self.IMP
                uh = u/10
                try:
                    self.QRS[i+j] += (rs + r_imp) * uh
                except Exception as e:
                    # print('[WARNING]:', e)
                    pass
            self.QRT[i] = np.sum([self.QRS[i], self.QRI[i], self.QRG[i]])  # 总计算径流
            # 计算下一个QRI
            self.QRI[i+1] = self.QRI[i] * self.KKSS + self.RI[i] * (1 - self.KKSS) * self.A / (3.6 * self.delta_t)
            self.QRG[i+1] = self.QRG[i] * self.KKG + self.RG[i] * (1 - self.KKG) * self.A / (3.6 * self.delta_t)

        # print(("|')<debug> ###汇流计算结束###\n')

    def matrix(self):
        matrix = np.hstack(([[i, ] for i in np.append(self.days, [0])],
                            [[i, ] for i in np.append(self.p, [0])],
                            [[i, ] for i in self.pe],
                            [[i, ] for i in self.e],
                            [[i, ] for i in self.w],
                            [[i, ] for i in self.QRT],
                            [[i, ] for i in np.append(self.q, [0])]))
        return matrix

    def go(self):
        self.runoff()
        self.water_source_split()
        self.concentrate()
        # print("<DEBUG> GO FINISH")

    # 计算离差平方和 sum of squares of deviations
    def ssd(self):
        _ave_q = np.average(self.q)
        _ave_qrt = np.average(self.QRT)
        _ssd = np.sum([(i-j)**2 for i, j in zip(self.QRT, self.q)])
        return _ssd

    def R_2(self):
        _ave = np.average(self.q)
        _r_2 = 1 - ((self.QRT[1:] - self.q)**2).sum() / ((self.q - _ave)**2).sum()
        return _r_2


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
        self.values = None
        self._args = {}

    def init_pop(self):
        for i, r in enumerate(self.ranges):
            self.pops[:, i] = np.array([random.uniform(r[0], r[1]) for j in range(self.pop_size)])
        return self.pops

    def evaluate(self):
        self.pops = self.pops[np.where(self.values > self.std)]
        return self.values[np.where(self.values > self.std)]

    def selection(self, weight):
        weight = list(weight)
        weight.sort()
        acc = list(accumulate(weight))
        rand = np.array([
            random.randint(0, 1) for i in range(len(self.pops))
        ])
        rand.sort()
        new_pop = []
        for i, r in enumerate(rand):
            for j in range(1, len(acc)+1):
                if acc[j-1] < r < acc[j]:
                    new_pop.append(self.pops[i])
        return new_pop

    def crossover(self, points=1):
        locs = [round(random.randint(0, 1), 1) for i in range(points)]
        locs = [int(i*self.chrom_length) for i in locs]
        loc = int(self.chrom_length/2)
        random.shuffle(self.pops)  # 打乱种群染色体
        # 单点交叉
        for i in range(int(len(self.pops)*self.p_cross)):
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

    def get_args(self):
        for p in range(self.pop_size):
            _args = {
                'WM': self.pops[p][0],
                'WUM': self.pops[p][1],
                'WLM': self.pops[p][2],
                'KC': self.pops[p][3],
                'C': self.pops[p][4],
                'B': self.pops[p][5],
                'IMP': 0.054,
                'FE': self.pops[p][6],
                'SM': self.pops[p][7],
                'EX': self.pops[p][8],
                'KG': self.pops[p][9],
                'KKG': self.pops[p][10],
                'KSS': self.pops[p][11],
                'KKSS': self.pops[p][12],
                'A': 3415,
                'UH': [0, 238.6, 63.8, 34.5, 20.6, 12.9, 8.3, 5.5, 3.6,
                       2.4, 1.6, 1.1, 0.8, 0.5, 0.4, 0.24, 0.16, 0.12,
                       0.08, 0.04, 0]
            }
            yield _args

    def set_values(self, _values):
        self.values = np.array(values)

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
    values = []

    conn = sqlite3.connect('baoriver.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM args")
    data_read = np.array(cur.fetchall(), dtype=float)
    UH = [0, 238.6, 63.8, 34.5, 20.6, 12.9, 8.3, 5.5, 3.6,
          2.4, 1.6, 1.1, 0.8, 0.5, 0.4, 0.24, 0.16, 0.12,
          0.08, 0.04, 0]
    data = data_read[0, 1:].flatten()
    args_tmp = {
        'WM': data[0],
        'WUM': data[1],
        'WLM': data[2],
        'KC': data[3],
        'C': data[4],
        'B': data[5],
        'IMP': data[6],
        'FE': data[7],
        'SM': data[8],
        'EX': data[9],
        'KG': data[10],
        'KKG': data[11],
        'KSS': data[12],
        'KKSS': data[13],
        'A': data[14],
        'UH': UH,
    }

    cur.execute("SELECT days,p,e,q FROM peq")
    data = np.array(cur.fetchall())
    data_dict = {
        'days': np.array(data[:, 0]),
        'p': np.array(data[:, 1], dtype=float),
        'e': np.array(data[:, 2], dtype=float),
        'q': np.array(data[:, 3], dtype=float)
    }

    # xinanjiang
    xinanjiang = Xinanjiang(_args=args_tmp, _data=data_dict)
    xinanjiang.go()
    print("模型效率系数：", xinanjiang.R_2())

    # args_tmp = {
    #     'WM': 0,
    #     'WUM': 0,
    #     'WLM': 0,
    #     'KC': 0,
    #     'C': 0,
    #     'B': 0,
    #     'IMP': 0,
    #     'FE': 0,
    #     'SM': 0,
    #     'EX': 0,
    #     'KG': 0,
    #     'KKG': 0,
    #     'KSS': 0,
    #     'KKSS': 0,
    #     'A': 0,
    #     'UH': 0,
    # }

    # K: 蒸发系数
    # IMP: 流域不透水系数
    # B: 流域蓄水容量曲线的方次
    # WM: 流域平均蓄水容量
    # WUM: 流域上层土壤平均蓄水容量
    # WLM: 流域下层土壤平均蓄水容量
    # C: 深层蒸散发折算系数
    # SM: 表层自由水蓄水容量
    # EX: 表层自由水蓄水容量曲线指数
    # KG: 地下水出流系数
    # KSS: 壤中流出流系数
    # KKG: 地下水消退系数
    # KKSS: 壤中流消退系数
    # UH: 单元流域上地面径流单位线

    # ga = GA(200, 13)
    # ga.init_pop()
    # xinanjiang = Xinanjiang(args_tmp, data_dict)
    # for g in range(ga.generation):
    #     for args in tqdm(ga.get_args()):
    #         xinanjiang.refresh_args(args)
    #         xinanjiang.go()
    #         # x = [i for i in range(len(data[:, 1]))]
    #         # plt.plot(x, xinanjiang.q, 'b--', label='q')
    #         # plt.plot(x, xinanjiang.QRT[:-1], 'r-', label='qrt')
    #         # plt.legend()
    #         # plt.show()
    #         # print("离差平方和：", xinanjiang.ssd())
    #         print("模型效率系数：", xinanjiang.R_2())
    #         values.append([xinanjiang.R_2()])  # 效率系数
    #     ga.set_values(values)  # 设置效率系数
    #     ev = ga.evaluate()  # 评价并淘汰
    #     ga.pops = ga.selection(ev)  # 选择新种群
    #     ga.crossover()
    #     ga.mutation()

    print("<DEBUG>")

    # result = xinanjiang.matrix()
    # df = pd.DataFrame(result)
    # df.to_excel("out.xls")
