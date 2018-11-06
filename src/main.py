import numpy as np
import pandas as pd
import os


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
    def __init__(self, args):
        super().__init__(args)
        self.kc = 0.9
        print("debug> kc == ", self.kc)
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
    def __init__(self, args):
        self.evaporation = Evaporation(args)
        self.dunne = Dunne(args)

    def calculate(self):
        for i in range(1461):
            self.evaporation.calculate(i, self.dunne)


if __name__ == '__main__':
    # 0.常量定义
    # 蒸发相关
    WM, WUM, WLM, WDM, C, B, FE = 115, 20, 25, 70, 0.16, 1.75, 0.8
    # 文件路径处理
    path = os.path.abspath('..')
    path = os.path.join(path, 'docs')
    path = os.path.join(path, 'baoRiverData.xlsx')
    excel_data = pd.read_excel(path, sheet_name='Sheet1')
    # 1.计算蒸发， 采用三层蒸发
    # eva_data = excel_data.loc[:, ['date', 'P', 'E']]
    ro_params = {
        'WM': WM,
        'WUM': WUM,
        'WLM': WLM,
        'WDM': WDM,
        'B': B,
        'C': C,
        'FE': FE,
        'data': excel_data,
    }
    # eva_calculator = RunOffGeneration(ro_params)
    # eva_calculator.Evaporation(ro_params).calculate()
    ro_calculator = RunOffCalculator(ro_params)
    ro_calculator.calculate()
    print(ro_calculator)
