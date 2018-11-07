import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class RunOffGeneration(object):
    def __init__(self, args):
        self.WM = args['WM']
        self.WUM = args['WUM']
        self.WLM = args['WLM']
        self.WDM = args['WDM']
        self.B = args['B']
        self.C = args['C']
        self.FE = args['FE']
        self.data_original = args['data']
        self.data = args['data']
        self._init_columns()

    def _init_columns(self):
        self.data = self.data.reindex(columns=['date',
                                               'P', 'pe',
                                               'E', 'ep', 'eu', 'el', 'ed', 'e',
                                               'wu', 'wl', 'wd', 'w',
                                               'R'])


class Dunne(RunOffGeneration):
    def __init__(self, args):
        super().__init__(args)
        self.W = self.WM * self.FE
        self.WMM = self._wmm()

    def _wmm(self):
        return self.WM * (1 + self.B)

    def _a(self, i):
        return self.WMM * (1 - (1 - self.data.loc[i, 'w'] / self.WM) ** (1 / (1 + self.B)))

    def _pe(self, i):
        pe = self.data.loc[i, 'P'] - self.data.loc[i, 'e']
        self.data.loc[i, 'pe'] = pe
        return pe

    def calculate(self, i):
        a = self.WMM * (1 - (1 - self.data.loc[i, 'w'] / self.WM) ** (1 / (1 + self.B)))
        PE = self._pe(i)
        if PE < 0:
            self.data.loc[i, 'R'] = 0
        else:
            if a + PE <= self.WMM:
                self.data.loc[i, 'R'] = PE + self.data.loc[i, 'w'] - self.WM + self.WM * (1 - (PE + a) / self.WMM) ** (self.B + 1)
            else:
                self.data.loc[i, 'R'] = PE - (self.WM - self.data.loc[i, 'w'])

    def sync_data(self, df):
        self.data.update(df)


class Evaporation(RunOffGeneration):
    def __init__(self, params):
        super().__init__(params)
        self.kc = params['KC']
        # print("debug> kc == ", self.kc)
        self.row = len(self.data.index)
        self.column = 10
        self._init_data()

    def _init_data(self):
        self.data.loc[0, 'wu'] = self.FE * self.WUM
        self.data.loc[0, 'wl'] = self.FE * self.WLM
        self.data.loc[0, 'wd'] = self.FE * self.WDM
        self.data.loc[0, 'w'] = self.data.loc[0, 'wu'] + self.data.loc[0, 'wl'] + self.data.loc[0, 'wd']
        self.data.loc[0, 'eu'] = 0
        self.data.loc[0, 'el'] = 0
        self.data.loc[0, 'ed'] = 0
        self.data.loc[0, 'e'] = 0
        self.data.loc[:, 'ep'] = self.ep()

    def ep(self):
        ep = [x * self.kc for x in self.data.loc[:, 'E']]
        return ep

    def eu_el_ed(self, i):
        wu, p, ep, wl, wd = self.data.loc[i, ['wu', 'P', 'ep', 'wl', 'wd']]
        if wu + p >= ep:
            self.data.loc[i, 'eu'] = ep
            self.data.loc[i, 'el'] = 0
            self.data.loc[i, 'ed'] = 0
        else:
            eu = wu + p
            self.data.loc[i, 'eu'] = eu
            if wl >= self.C * self.WLM:
                self.data.loc[i, 'el'] = ((ep - eu) * wl) / self.WLM
                self.data.loc[i, 'ed'] = 0
            elif self.C * (ep - eu) <= wl and wl < self.C * self.WLM:
                self.data.loc[i, 'el'] = self.C * (ep - eu)
                self.data.loc[i, 'ed'] = 0
            else:
                el = wl
                self.data.loc[i, 'el'] = el
                self.data.loc[i, 'ed'] = self.C * (ep - eu) - el
        self.data.loc[i, 'e'] = self.data.loc[i, 'eu'] + self.data.loc[i, 'el'] + self.data.loc[i, 'ed']
        # 计算W[j+1]

    def wu_wl_wd(self, j):
        wu, wl, wd, p, eu, el, ed, R = self.data.loc[j - 1, ['wu', 'wl', 'wd', 'P', 'eu', 'el', 'ed', 'R']]
        wu = wu + p - self.data.loc[j - 1, 'eu'] - R
        if wu < self.WUM:
            self.data.loc[j, 'wu'] = wu
            self.data.loc[j, 'wl'] = wl - el
            self.data.loc[j, 'wd'] = wd - ed
            self.data.loc[j, 'w'] = self.data.loc[j, 'wu'] + self.data.loc[j, 'wl'] + self.data.loc[j, 'wd']
            return 1
        self.data.loc[j, 'wu'] = self.WUM
        wl = wl - el + wu - self.WUM
        if wl < self.WLM:
            self.data.loc[j, 'wl'] = wl
            self.data.loc[j, 'wd'] = wd - ed
            self.data.loc[j, 'w'] = self.data.loc[j, 'wu'] + self.data.loc[j, 'wl'] + self.data.loc[j, 'wd']
            return 2
        self.data.loc[j, 'wl'] = self.WLM
        wd = wd - ed + wl - self.WLM
        if wd < self.WDM:
            self.data.loc[j, 'wd'] = wd
            self.data.loc[j, 'w'] = self.data.loc[j, 'wu'] + self.data.loc[j, 'wl'] + self.data.loc[j, 'wd']
            return 3
        self.data.loc[j, 'wd'] = self.WDM
        self.data.loc[j, 'w'] = self.data.loc[j, 'wu'] + self.data.loc[j, 'wl'] + self.data.loc[j, 'wd']

    def R(self, i, dunne):
        dunne.calculate(i)

    def sync_data(self, df):
        self.data.update(df)

    def calculate(self, i, dunne):
        self.eu_el_ed(i)
        dunne.sync_data(self.data)
        self.R(i, dunne)
        self.sync_data(dunne.data)
        self.wu_wl_wd(i + 1)


class RunOffCalculator(object):
    def __init__(self, params):
        self.evaporation = Evaporation(params)
        self.dunne = Dunne(params)
        self.row = len(params['data'].index)

    def calculate(self):
        for i in range(self.row):
            self.evaporation.calculate(i, self.dunne)


class WaterSourceSplit(object):
    def __init__(self, params):
        self.WM = params['WM']
        self.FE = params['FE']
        self.B = params['B']
        self.SM = params['SM']
        self.EX = params['EX']
        self.KG = params['KG']
        self.KKG = params['KKG']
        self.KSS = params['KSS']
        self.KKSS = params['KKSS']
        self.data = params['data']
        self._row = len(self.data.index)
        self._init_columns()

    def _init_columns(self):
        self.data = self.data.reindex(columns=['date',
                                               'P', 'pe',
                                               'E', 'ep', 'eu', 'el', 'ed', 'e',
                                               'wu', 'wl', 'wd', 'w',
                                               'FR', 'S',
                                               'R', 'RS', 'RI', 'RG'])
        self.data.loc[0, 'S'] = self.SM * self.FE

    def runoff(self, i):
        # s0 = self.data.loc[0, 'S']
        MS = self.SM * (1 + self.EX)
        pe, w, s, r = self.data.loc[i, ['pe', 'w', 'S', 'R']]
        # s0, fr0 = self.data.loc[i-1, ['S', 'FR']]
        if pe <= 0:
            fr = 1 - (1 - w / self.WM)**(self.B / (1 + self.B))
            self.data.loc[i, 'FR'] = fr
            # s = s0 * fr0 /
            self.data.loc[i, 'RS'] = 0
            self.data.loc[i, 'RI'] = s * self.KSS * fr
            self.data.loc[i, 'RG'] = s * self.KG * fr
            self.data.loc[i + 1, 'S'] = (1 - self.KSS - self.KG) * s
        else:
            fr = r / pe
            self.data.loc[i, 'FR'] = fr
            # AU = MS * (1 - (1 - (s0*(fr0/fr)) / self.SM)**(1 / (1 + self.EX)))
            AU = MS * (1 - (1 - s / self.SM)**(1 / (1 + self.EX)))
            if pe + AU < MS:
                # self.data.loc[i, 'RS'] = fr * (pe + s0 * fr0 / fr - self.SM + self.SM * (1 - (pe / AU) / MS) ** (
                #             self.EX + 1))
                tmp = self.SM * (1 - (pe + AU) / MS) ** (self.EX + 1)
                self.data.loc[i, 'RS'] = fr * (pe + s - self.SM + tmp)
                self.data.loc[i, 'RI'] = fr * self.KSS * (self.SM - tmp)
                self.data.loc[i, 'RG'] = fr * self.KG * (self.SM - tmp)
                self.data.loc[i + 1, 'S'] = (1 - self.KSS - self.KG) * (self.SM - tmp)
            else:
                self.data.loc[i, 'RS'] = fr * (pe - self.SM + s)
                self.data.loc[i, 'RI'] = fr * self.SM * self.KSS
                self.data.loc[i, 'RG'] = fr * self.KG * self.SM
                self.data.loc[i + 1, 'S'] = s * (1 - self.KSS * self.KG)

    def calculate(self):
        for i in range(self._row):
            self.runoff(i)


class Concentrate(object):
    def __init__(self, params):
        self.UH = params['UH']
        self.KKSS = params['KKSS']
        self.KKG = params['KKG']
        self.A = params['A']
        self.uh_len = len(self.UH)
        self.data = params['data']
        self.delta_t = 24
        self.row = len(self.data.index)
        self._init_columns()

    def _init_columns(self):
        self.data = self.data.reindex(columns=['date',
                                               'pe',
                                               'ep', 'eu', 'el', 'ed', 'e',
                                               'wu', 'wl', 'wd', 'w',
                                               'FR', 'S',
                                               'R', 'RS', 'RI', 'RG',
                                               'QRS', 'QRI', 'QRG', 'QRT'])
        self.data.loc[0, ['QRI', 'QRG']] = 0.5, 0.0
        self.data.loc[:, 'QRS'] = 0

    def calculate(self):
        for i in range(0, self.row):
            for j in range(self.uh_len):
                rs = self.data.loc[i, 'RS']
                uh = self.UH[j]/10
                self.data.loc[i+j, 'QRS'] = rs * uh
            QRI = self.data.loc[i, 'QRI']
            QRG = self.data.loc[i, 'QRG']
            RI = self.data.loc[i, 'RI']
            RG = self.data.loc[i, 'RG']
            self.data.loc[i+1, 'QRI'] = QRI * self.KKSS + RI * (1 - self.KKSS) * self.A / (3.6 * self.delta_t)
            self.data.loc[i+1, 'QRG'] = QRG * self.KKG + RG + (1 - self.KKG) * self.A / (3.6 * self.delta_t)
            self.data.loc[i, 'QRT'] = self.data.loc[i, 'QRS'] + self.data.loc[i, 'QRI'] + self.data.loc[i, 'QRG']


if __name__ == '__main__':
    # 文件路径处理
    path = os.path.abspath('..')
    path = os.path.join(path, 'docs')
    path = os.path.join(path, 'baoRiverData.xlsx')
    excel_data = pd.read_excel(path, sheet_name='Sheet1')
    plt.plot(excel_data.loc[:, 'date'], excel_data.loc[:, 'Q'])
    # 蒸发相关
    # 1.计算蒸发， 采用三层蒸发
    # 0.常量定义
    WM, WUM, WLM, WDM, C, B, FE, KC, IMP = 115, 20, 25, 70, 0.16, 1.75, 0.8, 1.04, 0.054
    ro_params = {
        'KC': KC,
        'WM': WM,
        'WUM': WUM,
        'WLM': WLM,
        'WDM': WDM,
        'B': B,
        'C': C,
        'FE': FE,
        'data': excel_data,
    }
    print('<debug> ###蒸发计算开始###\n')
    ro_calculator = RunOffCalculator(ro_params)
    ro_calculator.calculate()
    print('<debug> ###蒸发计算结束###\n')
    # 2.水源划分
    # 常量定义
    # SM --> 流域平均自由水蓄水容量
    # EX --> 自由水蓄量分布曲线指数
    # KG --> 自由水对地下水的日出流系数
    # KKG --> 地下水消退系数
    # KSS --> 自由水对壤中流的日出流系数
    # KKSS --> 壤中流消退系数
    SM, EX, KG, KKG, KSS, KKSS = 40, 1.9, 0.06, 0.995, 0.05, 0.885
    ws_params = {
        'WM': WM,
        'FE': FE,
        'B': B,
        'SM': SM,
        'EX': EX,
        'KG': KG,
        'KKG': KKG,
        'KSS': KSS,
        'KKSS': KKSS,
        'data': ro_calculator.evaporation.data,
    }
    print('<debug> ###水源划分计算开始###\n')
    ws_calculator = WaterSourceSplit(ws_params)
    ws_calculator.calculate()
    print('<debug> ###水源划分计算结束###\n')
    # 水源划分结束
    # 3.汇流开始
    # 定义常量
    A = 3415
    UH = [0, 238.6, 63.8, 34.5, 20.6, 12.9, 8.3, 5.5, 3.6,
          2.4, 1.6, 1.1, 0.8, 0.5, 0.4, 0.24, 0.16, 0.12,
          0.08, 0.04, 0]
    UH = np.array(UH)
    ct_params = {
        'UH': UH,
        'KKG': KKG,
        'A': A,
        'KKSS': KKSS,
        'data': ws_calculator.data,

    }
    print('<debug> ###汇流计算开始###\n')
    ct_calculator = Concentrate(ct_params)
    ct_calculator.calculate()
    print('<debug> ###汇流计算结束###\n')
    print('<debug> ###绘图开始###')
    plt.plot(ct_calculator.data.loc[:, 'date'], ct_calculator.data.loc[:, 'QRT'])
    plt.legend()
    plt.show()
    print('<debug> ###绘图结束###')


