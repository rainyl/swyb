import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import *
import sqlite3


# 产流计算
# 本类是产流计算类，只用于为蓄满产流类（Dunne）， 蒸法类（evaporation）初始化相关参数
class RunOffGeneration(object):
    def __init__(self, args):
        self.IMP = args['IMP']
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

    # 重新初始化列名
    def _init_columns(self):
        self.data = self.data.reindex(columns=['date',
                                               'P', 'pe',
                                               'E', 'ep', 'eu', 'el', 'ed', 'e',
                                               'wu', 'wl', 'wd', 'w',
                                               'R',
                                               'Q'])


# 蓄满产流计算
# 继承自runoffgeneration类
class Dunne(RunOffGeneration):
    def __init__(self, args):
        super().__init__(args)  # 调用父类初始化参数
        self.W = self.WM * self.FE
        self.WMM = self._wmm()

    # 内部计算WMM
    def _wmm(self):
        return self.WM * (1 + self.B)/(1 - self.IMP)

    def _a(self, i):
        return self.WMM * (1 - (1 - self.data.loc[i, 'w'] / self.WM) ** (1 / (1 + self.B)))

    def _pe(self, i):
        p = self.data.loc[i, 'P']
        pe = p * (1 - self.IMP) - self.data.loc[i, 'e']
        self.data.loc[i, 'pe'] = pe
        return pe

    # 计算过程总控制器
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

    # 同步数据
    # 由于将蒸法与产流分开两个类，因此二者之间的数据传递之前需要同步
    def sync_data(self, df):
        self.data.update(df)


# 蒸发计算类
class Evaporation(RunOffGeneration):
    def __init__(self, params):  # 调用父类初始化参数
        super().__init__(params)
        self.kc = params['KC']
        # print("debug> kc == ", self.kc)
        self.row = len(self.data.index)
        self.column = 10
        self._init_columns_()

    # 初始化列名
    def _init_columns_(self):
        self.data.loc[0, 'wu'] = self.FE * self.WUM
        self.data.loc[0, 'wl'] = self.FE * self.WLM
        self.data.loc[0, 'wd'] = self.FE * self.WDM
        self.data.loc[0, 'w'] = self.data.loc[0, 'wu'] + self.data.loc[0, 'wl'] + self.data.loc[0, 'wd']
        self.data.loc[0, 'eu'] = 0
        self.data.loc[0, 'el'] = 0
        self.data.loc[0, 'ed'] = 0
        self.data.loc[0, 'e'] = 0
        self.data.loc[:, 'ep'] = self.ep()

    # 计算Ep
    def ep(self):
        ep = [x * self.kc for x in self.data.loc[:, 'E']]
        return ep

    # 计算EU， EL， ED， E
    def eu_el_ed(self, i):
        wu, p, ep, wl, wd = self.data.loc[i, ['wu', 'P', 'ep', 'wl', 'wd']]
        p = p * (1 - self.IMP)  # 考虑不透水区域，其所占比例为IMP
        # 常规计算的使用分支的方法
        # if wu + p >= ep:
        #     self.data.loc[i, 'eu'] = ep
        #     self.data.loc[i, 'el'] = 0
        #     self.data.loc[i, 'ed'] = 0
        # else:
        #     eu = wu + p
        #     self.data.loc[i, 'eu'] = eu
        #     if wl >= self.C * self.WLM:
        #         self.data.loc[i, 'el'] = ((ep - eu) * wl) / self.WLM
        #         self.data.loc[i, 'ed'] = 0
        #     else:
        #         if wl >= self.C * (ep - eu):
        #             self.data.loc[i, 'el'] = self.C * (ep - eu)
        #             self.data.loc[i, 'ed'] = 0
        #         else:
        #             el = wl
        #             self.data.loc[i, 'el'] = el
        #             self.data.loc[i, 'ed'] = self.C * (ep - eu) - el
        #             if self.data.loc[i, 'ed'] > wd:
        #                 self.data.loc[i, 'ed'] = wd
        # 简化计算分支的方法
        # if wu + p < ep:
        #     eu = wu + p
        #     self.data.loc[i, 'eu'] = eu
        #     if wl < self.C * self.WLM:
        #         if self.C * (ep - eu) > wl:
        #             el = wl
        #             self.data.loc[i, 'el'] = el
        #             if self.data.loc[i, 'ed'] < wd:
        #                 self.data.loc[i, 'ed'] = self.C * (ep - eu) - el
        #             self.data.loc[i, 'ed'] = wd
        #         self.data.loc[i, 'el'] = self.C * (ep - eu)
        #         self.data.loc[i, 'ed'] = 0
        #     self.data.loc[i, 'el'] = ((ep - eu) * wl) / self.WLM
        #     self.data.loc[i, 'ed'] = 0
        # self.data.loc[i, 'eu'] = ep
        # self.data.loc[i, 'el'] = 0
        # self.data.loc[i, 'ed'] = 0
        self.data.loc[i, 'e'] = self.data.loc[i, 'eu'] + self.data.loc[i, 'el'] + self.data.loc[i, 'ed']  # 总E

    # 计算WU， WL， WD， W
    def wu_wl_wd(self, j):
        wu, wl, wd, p, eu, el, ed, R = self.data.loc[j - 1, ['wu', 'wl', 'wd', 'P', 'eu', 'el', 'ed', 'R']]
        wu = wu + p - self.data.loc[j - 1, 'eu'] - R
        # 同上，本处使用简化计算分支方法，减少代码缩进
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

    # 产流计算中间件
    def dunne_middle_ware(self, i, dunne):
        dunne.calculate(i)

    # 同步数据
    def sync_data(self, df):
        self.data.update(df)

    # 计算总控制器
    def calculate(self, i, dunne):
        self.eu_el_ed(i)
        dunne.sync_data(self.data)
        self.dunne_middle_ware(i, dunne)
        self.sync_data(dunne.data)
        self.wu_wl_wd(i + 1)


# 产流计算总控制器
class RunOffCalculator(object):
    def __init__(self, params):
        self.evaporation = Evaporation(params)
        self.dunne = Dunne(params)
        self.row = len(params['data'].index)

    def calculate(self):
        for i in tqdm(range(self.row)):
            self.evaporation.calculate(i, self.dunne)


# 水源划分类
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

    # 初始化列名
    def _init_columns(self):
        self.data = self.data.reindex(columns=['date',
                                               'P', 'pe',
                                               'E', 'ep', 'eu', 'el', 'ed', 'e',
                                               'wu', 'wl', 'wd', 'w',
                                               'FR', 'S',
                                               'R', 'RS', 'RI', 'RG',
                                               'Q'])
        self.data.loc[0, 'S'] = self.SM * self.FE

    # 水源划分
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
                self.data.loc[i + 1, 'S'] = self.SM * (1 - self.KSS * self.KG)

    def calculate(self):
        for i in tqdm(range(self._row)):
            self.runoff(i)


# 汇流计算类
class Concentrate(object):
    def __init__(self, params):
        self.IMP = params['IMP']
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
                                               'P', 'pe',
                                               'ep', 'eu', 'el', 'ed', 'e',
                                               'wu', 'wl', 'wd', 'w',
                                               'FR', 'S',
                                               'R', 'RS', 'RI', 'RG',
                                               'QRS', 'QRI', 'QRG', 'QRT', 'Q'])
        self.data.loc[0, ['QRI', 'QRG']] = 0.5, 0.0
        self.data.loc[:, 'QRS'] = 0

    # 主计算方法
    def calculate(self):
        for i in tqdm(range(self.row)):
            for j in range(self.uh_len):
                rs = self.data.loc[i, 'RS']
                r_imp = self.data.loc[i, 'P'] * self.IMP
                uh = self.UH[j]/10
                try:
                    self.data.loc[i+j, 'QRS'] += (rs + r_imp) * uh
                except Exception as e:
                    print('[WARNING]:', e)
            QRI = self.data.loc[i, 'QRI']  # 壤中流
            QRG = self.data.loc[i, 'QRG']  # 地下径流
            RI = self.data.loc[i, 'RI']  # 壤中流产流
            RG = self.data.loc[i, 'RG']  # 地下径流产流
            self.data.loc[i, 'QRT'] = self.data.loc[i, 'QRS'] + self.data.loc[i, 'QRI'] + self.data.loc[i, 'QRG']  # 总计算径流
            self.data.loc[i+1, 'QRI'] = QRI * self.KKSS + RI * (1 - self.KKSS) * self.A / (3.6 * self.delta_t)  # 计算下一个QRI
            self.data.loc[i+1, 'QRG'] = QRG * self.KKG + RG * (1 - self.KKG) * self.A / (3.6 * self.delta_t)


# 洪水场次分割
class FloodSplit(object):
    def __init__(self, params):
        self.flood_data = params['flood_data']
        self.excel_data = params['excel_data']
        self.row_flood_data = len(self.flood_data.index)
        self.flood_start_index = []
        self.flood_stop_index = []

    # 主分割方法，使用索引号定位
    def split_flood_data(self):
        # 1. 分割洪号、起止时间
        # 起 = '19' + 洪号前两个 + '-' + 起时间前两个 + '-' + 起时间后两个
        # 止 = '19' + 洪号前两个 + '-' + 止时间前两个 + '-' + 止时间后两个
        flood_num = np.array(self.flood_data.loc[:, 'flood_num'])
        flood_year = []
        flood_start_stop = np.array(self.flood_data.loc[:, 'start_stop'])
        flood_one_start = []
        flood_one_stop = []
        flood_start = []
        flood_stop = []
        for flood in flood_num:
            flood_year.append('19' + str(flood)[0:2])
        for flood in flood_start_stop:
            flood_one_start.append(str(flood)[0:2] + '-' + str(flood)[3:5])
            flood_one_stop.append(str(flood)[6:8] + '-' + str(flood)[9:11])
        for year, start, stop in zip(flood_year, flood_one_start, flood_one_stop):
            flood_start.append(year + '-' + start)
            flood_stop.append(year + '-' + stop)
            print(year + '-' + start + '-->' + year + '-' + stop)
        # 2. 定位起止时间索引号
        excel_data_date = self.excel_data.loc[:, 'date']
        excel_data_date = [str(x.date()) for x in excel_data_date]
        for start, stop in zip(flood_start, flood_stop):
            self.flood_start_index.append(excel_data_date.index(start))
            self.flood_stop_index.append(excel_data_date.index(stop))

    # 计算洪水总量
    def sum_flood(self):
        for start, stop, row in zip(self.flood_start_index, self.flood_stop_index, range(self.row_flood_data)):
            flood_measured = np.array(self.excel_data.loc[start:stop, 'Q'])
            flood_simulated = np.array(self.excel_data.loc[start:stop, 'QRT'])
            flood_p = np.array(self.excel_data.loc[start:stop, 'P'])
            # 确定性系数计算
            ave = flood_measured.sum() / len(flood_measured)  # 实测均值
            up = np.array([(yc - y0) ** 2 for yc, y0 in zip(flood_simulated, flood_measured)]).sum()  # 分子
            down = np.array([(y0 - ave) ** 2 for y0 in flood_measured]).sum()  # 分母
            R_2 = 1 - up / down  # 确定性系数
            self.flood_data.loc[row, 'R^2'] = R_2  # 赋值给相应dataframe
            # 洪水总量计算
            self.flood_data.loc[row, 'sum_measured_flood'] = flood_measured.sum()
            self.flood_data.loc[row, 'sum_simulated_flood'] = flood_simulated.sum()
            self.flood_data.loc[row, 'P'] = flood_p.sum()

    # 洪峰计算
    def peak(self):
        peak_measured = []
        peak_simulated = []
        for start, stop, row in zip(self.flood_start_index, self.flood_stop_index, range(self.row_flood_data)):
            # 实测洪峰
            flood_measured = self.excel_data.loc[start:stop, ['date', 'Q']]
            flood_measured.index = range(len(flood_measured))
            max_ = flood_measured.loc[:, 'Q'].max()
            self.flood_data.loc[row, 'measured_peak'] = max_
            index = list(flood_measured.loc[:, 'Q']).index(max_)  # 洪峰索引
            date = flood_measured.loc[index, 'date']  # 峰现时间
            self.flood_data.loc[row, 'measured_peak_time'] = date
            peak_measured.append((max_, date))
            peak_measured.append((flood_measured.max(), ))
            # 计算洪峰
            flood_simulated = self.excel_data.loc[start:stop, ['date', 'QRT']]
            flood_simulated.index = range(len(flood_simulated))
            max_ = flood_simulated.loc[:, 'QRT'].max()
            self.flood_data.loc[row, 'simulated_peak'] = max_
            index = list(flood_simulated.loc[:, 'QRT']).index(max_)
            date = flood_simulated.loc[index, 'date']
            self.flood_data.loc[row, 'simulated_peak_time'] = date
            peak_simulated.append((max_, date))
        self.__setattr__('peak_measured', peak_measured)
        self.__setattr__('peak_simulated', peak_simulated)
        return peak_measured, peak_simulated

    # 主计算控制器
    def calculate(self):
        self.split_flood_data()
        self.sum_flood()
        self.peak()


# 分析类
class Analyst(object):
    def __init__(self, params):
        self.A = params['A']
        self.pr_data = params['pr_data']
        self.flood_data_all = params['flood_data_all']
        self.flood_data = params['flood_data']
        self._init_columns()

    def _init_columns(self):
        self.flood_data_all.reindex(columns=['NO',
                                             'flood_num',
                                             'P',
                                             'RO',
                                             'RC',
                                             'abs_error',
                                             'relative_error',
                                             'R^2'])

    # 误差计算
    # 不同于上述使用定位索引号的方法进行索引
    # 本处使用直接更改索引号为洪号，再使用洪号直接定位数据
    def error(self):
        flood_num = self.flood_data.loc[:, 'flood_num']
        self.flood_data_all.index = list(self.flood_data_all.loc[:, 'flood_num'])
        self.flood_data.index = flood_num
        ro_ = self.flood_data.loc[:, 'sum_measured_flood']
        rc_ = self.flood_data.loc[:, 'sum_simulated_flood']
        ro_ = [r * 3600 * 24 / (self.A * 10 ** 3) for r in ro_]
        rc_ = [r * 3600 * 24 / (self.A * 10 ** 3) for r in rc_]
        for flood, ro, rc in zip(flood_num, ro_, rc_):
            p, r_2 = self.flood_data.loc[flood, ['P', 'R^2']]
            self.flood_data_all.loc[flood, 'abs_error'] = rc - ro  # 绝对误差
            self.flood_data_all.loc[flood, 'P'] = p  # 场次总净雨
            self.flood_data_all.loc[flood, 'RO'] = ro  # 场次实测总径流
            self.flood_data_all.loc[flood, 'RC'] = rc  # 场次计算总径流
            self.flood_data_all.loc[flood, 'R^2'] = r_2  # 确定性系数
            self.flood_data_all.loc[flood, 'relative_error'] = 100 * (rc - ro) / ro  # 相对误差
        self.flood_data_all.index = range(len(self.flood_data_all.index))  # 恢复index索引
        self.flood_data.index = range(len(self.flood_data.index))  # 恢复index索引

    # 分析控制器
    def analyse(self):
        self.error()


# 主函数
if __name__ == '__main__':
    # 文件路径处理
    WORKSPACE = os.path.abspath('.')
    path_docs = os.path.join(WORKSPACE, 'docs')  # 数据路径
    path_save_to = os.path.join(path_docs, 'baoRiverDataCal.xlsx')  # 存储路径
    path_source_data = os.path.join(path_docs, 'baoRiverData.xlsx')

    data = np.array(pd.read_excel(path_source_data, 0))
    for i, d in enumerate(data[:, 0]):
        data[i, 0] = d.strftime('%Y-%m-%d')
    db = sqlite3.connect('baoriver.db')
    cur = db.cursor()
    cur.executemany("INSERT INTO peq(days,p,e,q) VALUES(?,?,?,?);", data)
    db.commit()
    db.close()

    excel_data = pd.read_excel(path_source_data, sheet_name='Sheet1')
    flood_data = pd.read_excel(path_source_data, sheet_name='Sheet2')
    flood_data_all = pd.read_excel(path_source_data, sheet_name='Sheet3')  # 读取原始数据到dataframe中
    plt.plot(excel_data.loc[:, 'date'], excel_data.loc[:, 'Q'])

    # 蒸发相关
    # 1.计算蒸发， 采用三层蒸发
    # 0.常量定义
    WM, WUM, WLM, WDM, C, B, FE, KC, IMP = 115, 20, 25, 70, 0.16, 1.75, 0.8, 1.04, 0.054
    # 参数定义，用字典传递
    ro_params = {
        'IMP': IMP,
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
    ro_calculator = RunOffCalculator(ro_params)  # 实例化产流计算对象
    ro_calculator.calculate()  # 调用上述对象的主计算控制器进行计算
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
    ws_calculator = WaterSourceSplit(ws_params)  # 实例化水源划分对象
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
        'IMP': IMP,
        'UH': UH,
        'KKG': KKG,
        'A': A,
        'KKSS': KKSS,
        'data': ws_calculator.data,

    }
    print('<debug> ###汇流计算开始###\n')
    # 4.汇流计算
    # 定义常量
    ct_calculator = Concentrate(ct_params)  # 实例化汇流计算对象
    ct_calculator.calculate()
    print('<debug> ###汇流计算结束###\n')
    print('<debug> ###洪水分割计算开始###\n')
    fs_params = {
        'excel_data': ct_calculator.data,
        'flood_data': flood_data,
    }
    fs_calculator = FloodSplit(fs_params)  # 实例化洪水分割对象
    fs_calculator.calculate()
    print('<debug> ###洪水分割计算结束###\n')
    print('<debug> ###分析数据开始###')
    ana_params = {
        'A': A,
        'pr_data': ct_calculator.data,
        'flood_data_all': flood_data_all,
        'flood_data': flood_data,
    }
    analyst = Analyst(ana_params)  # 实例化分析对象
    analyst.analyse()
    print('<debug> ###分析数据结束###')
    print('<debug> ###绘图开始###')
    plt.plot(ct_calculator.data.loc[:, 'date'], ct_calculator.data.loc[:, 'QRT'])
    plt.legend()
    plt.show()
    print('<debug> ###绘图结束###')
    print('<debug> ###导出数据开始###')
    writer = pd.ExcelWriter(path_save_to)
    try:
        ct_calculator.data.to_excel(writer, sheet_name='蒸发-产流-汇流计算表')
        fs_calculator.flood_data.to_excel(writer, sheet_name='次洪统计表')
        analyst.flood_data_all.to_excel(writer, sheet_name='次洪总表与分析')
    except Exception as e:
        print(e)
    writer.save()
    writer.close()
    print('<debug> ###导出数据结束###')
