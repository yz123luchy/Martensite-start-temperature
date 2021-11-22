import math
import numpy as np
import pandas as pd
from scipy import stats
import joblib

#Ten fold cross validation, Train-Test-Split
from sklearn.model_selection import cross_val_score,cross_val_predict, KFold, train_test_split as TTS
#Error metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
#Plot
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
plt.rc('font',family='Times New Roman')
#LightGBM model
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
#Model interpretability analysis
from pdpbox import pdp, get_dataset, info_plots

#Display all columns
pd.set_option('display.max_columns', None)
#Display all rows
pd.set_option('display.max_rows', None)

df_va = pd.read_csv('./data/validation/Bohemen_validation.csv')
df_va['Ms (K)'] = df_va['exp_Ms'] + 273.15
df_va['cal_Ms'] = df_va['calc_Ms'] + 273.15
df_va['Jmatpro_1'] = df_va['Jmatpro'] + 273.15
print(df_va)

df_4 = pd.read_csv('./data/df_4.csv')
X_ele_fea = df_4[['C','Si','Mn','Ni','Cr','Mo','V','Cu','W',\
                  'Al','Ti','Nb','N','Co','B','P','S', '∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']]
Y_ele_fea = df_4['Ms (K)']
reg_ele_fea = LGBMRegressor(n_estimators = 1000,random_state = 123).fit(X_ele_fea  , Y_ele_fea)


def ghosh(df):
    Fe_w = 100 - df['C'] - df['Mn'] - df['Si'] - df['Cr'] - \
           df['Ni'] - df['Mo'] - df['V'] - df['Co'] - \
           df['Al'] - df['W'] - df['Cu'] - df['Nb'] - \
           df['Ti'] - df['B'] - df['N'] - df['P'] - df['S']
    Fe_mol = Fe_w / 55.845

    C_mol = df['C'] / 12.017
    MN_mol = df['Mn'] / 54.938
    Si_mol = df['Si'] / 28.085
    CR_mol = df['Cr'] / 51.996
    NI_mol = df['Ni'] / 58.693
    MO_mol = df['Mo'] / 95.94
    V_mol = df['V'] / 50.941
    Co_mol = df['Co'] / 58.933
    AL_mol = df['Al'] / 26.981
    W_mol = df['W'] / 183.84
    CU_mol = df['Cu'] / 63.546
    NB_mol = df['Nb'] / 92.906
    TI_mol = df['Ti'] / 47.867
    B_mol = df['B'] / 10.811
    N_mol = df['N'] / 14.006
    P_mol = df['P'] / 30.973
    S_mol = df['S'] / 32.065

    ele_mol_all = Fe_mol + C_mol + MN_mol + Si_mol + CR_mol + NI_mol + MO_mol \
                  + V_mol + Co_mol + AL_mol + W_mol + CU_mol + NB_mol + TI_mol + B_mol + N_mol + P_mol + S_mol

    Fe_mol_x = Fe_mol / ele_mol_all
    C_mol_x = C_mol / ele_mol_all
    MN_mol_x = MN_mol / ele_mol_all
    Si_mol_x = Si_mol / ele_mol_all
    CR_mol_x = CR_mol / ele_mol_all
    NI_mol_x = NI_mol / ele_mol_all
    MO_mol_x = MO_mol / ele_mol_all
    V_mol_x = V_mol / ele_mol_all
    Co_mol_x = Co_mol / ele_mol_all
    AL_mol_x = AL_mol / ele_mol_all
    W_mol_x = W_mol / ele_mol_all
    CU_mol_x = CU_mol / ele_mol_all
    NB_mol_x = NB_mol / ele_mol_all
    TI_mol_x = TI_mol / ele_mol_all
    B_mol_x = B_mol / ele_mol_all
    N_mol_x = N_mol / ele_mol_all
    P_mol_x = P_mol / ele_mol_all
    S_mol_x = S_mol / ele_mol_all

    ghosh_1 = np.sqrt(np.power(4009 * np.sqrt(C_mol_x), 2) + np.power(3097 * np.sqrt(N_mol_x), 2))
    ghosh_2 = np.sqrt(np.power(1868 * np.sqrt(CR_mol_x), 2) + np.power(1980 * np.sqrt(MN_mol_x), 2) + np.power(
        1418 * np.sqrt(MO_mol_x), 2) + np.power(1653 * np.sqrt(NB_mol_x), 2) + np.power(1879 * np.sqrt(Si_mol_x),
                                                                                        2) + np.power(
        1473 * np.sqrt(TI_mol_x), 2) + np.power(1618 * np.sqrt(V_mol_x), 2))
    ghosh_3 = np.sqrt(
        np.power(280 * np.sqrt(AL_mol_x), 2) + np.power(752 * np.sqrt(CU_mol_x), 2) + np.power(172 * np.sqrt(NI_mol_x),
                                                                                               2) + np.power(
            714 * np.sqrt(W_mol_x), 2))
    ghosh_4 = -352 * np.sqrt(Co_mol_x)

    Ghosh = ghosh_1 + ghosh_2 + ghosh_3 + ghosh_4

    df['∆Gc'] = Ghosh

    return df

#Atomic characteristics and lattice parameters
C_r = 0.77
Si_r = 1.11
Mn_r = 1.39
Ni_r = 1.21
Cr_r = 1.27
Mo_r = 1.45
V_r = 1.25
Cu_r = 1.38
W_r = 1.46
Al_r = 1.18
Ti_r = 1.36
Nb_r = 1.37
N_r = 0.75
Co_r = 1.26
B_r = 0.82
P_r = 1.06
S_r = 1.02
Fe_r = 1.25

#The quantity of Valence electron  of the alloy elements or iron
C_ve  = 4
Si_ve = 4
Mn_ve = 7
Ni_ve = 10
Cr_ve = 6
Mo_ve = 6
V_ve  = 5
Cu_ve = 11
W_ve  = 6
Al_ve = 3
Ti_ve = 4
Nb_ve = 5
N_ve  = 5
Co_ve = 9
B_ve  = 3
P_ve  = 5
S_ve  = 6
Fe_ve = 8

#electronegativity of the alloy elements
C_eg  = 2.55
Si_eg = 1.9
Mn_eg = 1.55
Ni_eg = 1.91
Cr_eg = 1.66
Mo_eg = 2.16
V_eg  = 1.63
Cu_eg = 1.90
W_eg  = 2.36
Al_eg = 1.61
Ti_eg = 1.54
Nb_eg = 1.6
N_eg  = 3.04
Co_eg = 1.88
B_eg  = 2.04
P_eg  = 2.19
S_eg  = 2.58
Fe_eg = 1.83


# mole fraction of the alloy elements
def Atom_fraction(df):
    Fe_w = 100 - df['C'] - df['Mn'] - df['Si'] - df['Cr'] - df['Ni'] - df['Mo'] - df['V'] - \
           df['Co'] - df['Al'] - df['W'] - df['Cu'] - df['Nb'] - df['Ti'] - df['B'] - df['N'] - df['P'] - df['S']
    Fe_mol = Fe_w / 55.845

    C_mol = df['C'] / 12.017
    MN_mol = df['Mn'] / 54.938
    Si_mol = df['Si'] / 28.085
    CR_mol = df['Cr'] / 51.996
    NI_mol = df['Ni'] / 58.693
    MO_mol = df['Mo'] / 95.94
    V_mol = df['V'] / 50.941
    Co_mol = df['Co'] / 58.933
    AL_mol = df['Al'] / 26.981
    W_mol = df['W'] / 183.84
    CU_mol = df['Cu'] / 63.546
    NB_mol = df['Nb'] / 92.906
    TI_mol = df['Ti'] / 47.867
    B_mol = df['B'] / 10.811
    N_mol = df['N'] / 14.006
    P_mol = df['P'] / 30.973
    S_mol = df['S'] / 32.065

    ele_mol_all = Fe_mol + C_mol + MN_mol + Si_mol + CR_mol + NI_mol + MO_mol + V_mol + Co_mol + AL_mol + W_mol + CU_mol + NB_mol + TI_mol + B_mol + N_mol + P_mol + S_mol
    Fe_mol_x = Fe_mol / ele_mol_all
    C_mol_x = C_mol / ele_mol_all
    MN_mol_x = MN_mol / ele_mol_all
    Si_mol_x = Si_mol / ele_mol_all
    CR_mol_x = CR_mol / ele_mol_all
    NI_mol_x = NI_mol / ele_mol_all
    MO_mol_x = MO_mol / ele_mol_all
    V_mol_x = V_mol / ele_mol_all
    Co_mol_x = Co_mol / ele_mol_all
    AL_mol_x = AL_mol / ele_mol_all
    W_mol_x = W_mol / ele_mol_all
    CU_mol_x = CU_mol / ele_mol_all
    NB_mol_x = NB_mol / ele_mol_all
    TI_mol_x = TI_mol / ele_mol_all
    B_mol_x = B_mol / ele_mol_all
    N_mol_x = N_mol / ele_mol_all
    P_mol_x = P_mol / ele_mol_all
    S_mol_x = S_mol / ele_mol_all

    df['Fe_mol'] = Fe_mol_x
    df['C_mol'] = C_mol_x
    df['Mn_mol'] = MN_mol_x
    df['Si_mol'] = Si_mol_x
    df['Cr_mol'] = CR_mol_x
    df['Ni_mol'] = NI_mol_x
    df['Mo_mol'] = MO_mol_x
    df['V_mol'] = V_mol_x
    df['Co_mol'] = Co_mol_x
    df['Al_mol'] = AL_mol_x
    df['W_mol'] = W_mol_x
    df['Cu_mol'] = CU_mol_x
    df['Nb_mol'] = NB_mol_x
    df['Ti_mol'] = TI_mol_x
    df['B_mol'] = B_mol_x
    df['N_mol'] = N_mol_x
    df['P_mol'] = P_mol_x
    df['S_mol'] = S_mol_x

    return df


#总原子半径
def Atom_R(df):
    df['Sum_Atom_R'] = C_r  * df['C_mol'] + \
           Si_r * df['Si_mol']+ \
           Mn_r * df['Mn_mol']+\
           Ni_r * df['Ni_mol']+\
           Cr_r * df['Cr_mol']+\
           Mo_r * df['Mo_mol']+\
           V_r  * df['V_mol'] +\
           Cu_r * df['Cu_mol']+\
           W_r  * df['W_mol'] +\
           Al_r * df['Al_mol']+\
           Ti_r * df['Ti_mol']+\
           Nb_r * df['Nb_mol']+\
           N_r  * df['N_mol'] +\
           Co_r * df['Co_mol']+\
           B_r  * df['B_mol'] +\
           P_r  * df['P_mol'] +\
           S_r  * df['S_mol'] +\
           Fe_r * df['Fe_mol']
    return df


#原子半径差(铁基)
import math
def Atom_difference(df):
    df['Atom_diff(Fe)_1'] = math.pow(1- C_r  / Fe_r, 2) * df['C_mol']  +\
                math.pow(1- Si_r / Fe_r, 2) * df['Si_mol'] +\
                math.pow(1- Mn_r / Fe_r, 2) * df['Mn_mol'] +\
                math.pow(1- Ni_r / Fe_r, 2) * df['Ni_mol'] +\
                math.pow(1- Cr_r / Fe_r, 2) * df['Cr_mol'] +\
                math.pow(1- Mo_r / Fe_r, 2) * df['Mo_mol'] +\
                math.pow(1- V_r  / Fe_r, 2) * df['V_mol']  +\
                math.pow(1- Cu_r / Fe_r, 2) * df['Cu_mol'] +\
                math.pow(1- W_r  / Fe_r, 2) * df['W_mol']  +\
                math.pow(1- Al_r / Fe_r, 2) * df['Al_mol'] +\
                math.pow(1- Ti_r / Fe_r, 2) * df['Ti_mol'] +\
                math.pow(1- Nb_r / Fe_r, 2) * df['Nb_mol'] +\
                math.pow(1- N_r  / Fe_r, 2) * df['N_mol']  +\
                math.pow(1- Co_r / Fe_r, 2) * df['Co_mol'] +\
                math.pow(1- B_r  / Fe_r, 2) * df['B_mol']  +\
                math.pow(1- P_r  / Fe_r, 2) * df['P_mol']  +\
                math.pow(1- S_r  / Fe_r, 2) * df['S_mol']  +\
                math.pow(1- Fe_r / Fe_r, 2) * df['Fe_mol']

    df['Atom_diff(Fe)'] = df['Atom_diff(Fe)_1'].apply(np.sqrt)
    return df


#总价电子数
def All_VEC_Cal(df):
    df['Sum_VEN'] = C_ve  * df['C_mol'] +\
           Si_ve * df['Si_mol']+ \
           Mn_ve * df['Mn_mol']+\
           Ni_ve * df['Ni_mol']+\
           Cr_ve * df['Cr_mol']+\
           Mo_ve * df['Mo_mol']+\
           V_ve  * df['V_mol'] +\
           Cu_ve * df['Cu_mol']+\
           W_ve  * df['W_mol'] +\
           Al_ve * df['Al_mol']+\
           Ti_ve * df['Ti_mol']+\
           Nb_ve * df['Nb_mol']+\
           N_ve  * df['N_mol'] +\
           Co_ve * df['Co_mol']+\
           B_ve  * df['B_mol'] +\
           P_ve  * df['P_mol'] +\
           S_ve  * df['S_mol'] +\
           Fe_ve * df['Fe_mol']
    return df

#价电子数差(铁基)
def VEC_Fe_Cal(df):
    df['VEC_Fe_1'] = math.pow(1- C_ve   / Fe_ve, 2) * df['C_mol']  +\
                    math.pow(1- Si_ve  / Fe_ve, 2) * df['Si_mol'] +\
                    math.pow(1- Mn_ve  / Fe_ve, 2) * df['Mn_mol'] +\
                    math.pow(1- Ni_ve  / Fe_ve, 2) * df['Ni_mol'] +\
                    math.pow(1- Cr_ve  / Fe_ve, 2) * df['Cr_mol'] +\
                    math.pow(1- Mo_ve  / Fe_ve, 2) * df['Mo_mol'] +\
                    math.pow(1- V_ve   / Fe_ve, 2) * df['V_mol']  +\
                    math.pow(1- Cu_ve  / Fe_ve, 2) * df['Cu_mol'] +\
                    math.pow(1- W_ve   / Fe_ve, 2) * df['W_mol']  +\
                    math.pow(1- Al_ve  / Fe_ve, 2) * df['Al_mol'] +\
                    math.pow(1- Ti_ve  / Fe_ve, 2) * df['Ti_mol'] +\
                    math.pow(1- Nb_ve  / Fe_ve, 2) * df['Nb_mol'] +\
                    math.pow(1- N_ve   / Fe_ve, 2) * df['N_mol']  +\
                    math.pow(1- Co_ve  / Fe_ve, 2) * df['Co_mol'] +\
                    math.pow(1- B_ve   / Fe_ve, 2) * df['B_mol']  +\
                    math.pow(1- P_ve   / Fe_ve, 2) * df['P_mol']  +\
                    math.pow(1- S_ve   / Fe_ve, 2) * df['S_mol']  +\
                    math.pow(1- Fe_ve  / Fe_ve, 2) * df['Fe_mol']

    df['VEN_Fe'] = df['VEC_Fe_1'].apply(np.sqrt)
    return df

#价电子数差(碳基）
def VEC_C_Cal(df):
    df['VEC_C_1'] = math.pow(1- C_ve   / C_ve, 2) * df['C_mol']  +\
             math.pow(1- Si_ve  / C_ve, 2) * df['Si_mol'] +\
             math.pow(1- Mn_ve  / C_ve, 2) * df['Mn_mol'] +\
             math.pow(1- Ni_ve  / C_ve, 2) * df['Ni_mol'] +\
             math.pow(1- Cr_ve  / C_ve, 2) * df['Cr_mol'] +\
             math.pow(1- Mo_ve  / C_ve, 2) * df['Mo_mol'] +\
             math.pow(1- V_ve   / C_ve, 2) * df['V_mol']  +\
             math.pow(1- Cu_ve  / C_ve, 2) * df['Cu_mol'] +\
             math.pow(1- W_ve   / C_ve, 2) * df['W_mol']  +\
             math.pow(1- Al_ve  / C_ve, 2) * df['Al_mol'] +\
             math.pow(1- Ti_ve  / C_ve, 2) * df['Ti_mol'] +\
             math.pow(1- Nb_ve  / C_ve, 2) * df['Nb_mol'] +\
             math.pow(1- N_ve   / C_ve, 2) * df['N_mol']  +\
             math.pow(1- Co_ve  / C_ve, 2) * df['Co_mol'] +\
             math.pow(1- B_ve   / C_ve, 2) * df['B_mol']  +\
             math.pow(1- P_ve   / C_ve, 2) * df['P_mol']  +\
             math.pow(1- S_ve   / C_ve, 2) * df['S_mol']  +\
             math.pow(1- Fe_ve  / C_ve, 2) * df['Fe_mol']

    df['VEN_C'] = df['VEC_C_1'].apply(np.sqrt)
    return df

#总鲍林电负性
def Ele_Gativity(df):
    df['Sum_EN'] = C_eg  * df['C_mol'] + \
                   Si_eg * df['Si_mol']+\
                   Mn_eg * df['Mn_mol']+\
                   Ni_eg * df['Ni_mol']+\
                   Cr_eg * df['Cr_mol']+\
                   Mo_eg * df['Mo_mol']+\
                   V_eg  * df['V_mol'] +\
                   Cu_eg * df['Cu_mol']+\
                   W_eg  * df['W_mol'] +\
                   Al_eg * df['Al_mol']+\
                   Ti_eg * df['Ti_mol']+\
                   Nb_eg * df['Nb_mol']+\
                   N_eg  * df['N_mol'] +\
                   Co_eg * df['Co_mol']+\
                   B_eg  * df['B_mol'] +\
                   P_eg  * df['P_mol'] +\
                   S_eg  * df['S_mol'] +\
                   Fe_eg * df['Fe_mol']
    return df



#电负性差(铁基)
def eg_Fe(df):
    df['Ele_Gativity_Fe_1'] = math.pow(1- C_eg   / Fe_eg, 2) * df['C_mol']  +\
                    math.pow(1- Si_eg  / Fe_eg, 2) * df['Si_mol'] +\
                    math.pow(1- Mn_eg  / Fe_eg, 2) * df['Mn_mol'] +\
                    math.pow(1- Ni_eg  / Fe_eg, 2) * df['Ni_mol'] +\
                    math.pow(1- Cr_eg  / Fe_eg, 2) * df['Cr_mol'] +\
                    math.pow(1- Mo_eg  / Fe_eg, 2) * df['Mo_mol'] +\
                    math.pow(1- V_eg   / Fe_eg, 2) * df['V_mol']  +\
                    math.pow(1- Cu_eg  / Fe_eg, 2) * df['Cu_mol'] +\
                    math.pow(1- W_eg   / Fe_eg, 2) * df['W_mol']  +\
                    math.pow(1- Al_eg  / Fe_eg, 2) * df['Al_mol'] +\
                    math.pow(1- Ti_eg  / Fe_eg, 2) * df['Ti_mol'] +\
                    math.pow(1- Nb_eg  / Fe_eg, 2) * df['Nb_mol'] +\
                    math.pow(1- N_eg   / Fe_eg, 2) * df['N_mol']  +\
                    math.pow(1- Co_eg  / Fe_eg, 2) * df['Co_mol'] +\
                    math.pow(1- B_eg   / Fe_eg, 2) * df['B_mol']  +\
                    math.pow(1- P_eg   / Fe_eg, 2) * df['P_mol']  +\
                    math.pow(1- S_eg   / Fe_eg, 2) * df['S_mol']  +\
                    math.pow(1- Fe_eg  / Fe_eg, 2) * df['Fe_mol']

    df['EN_Fe'] = df['Ele_Gativity_Fe_1'].apply(np.sqrt)
    return df



#电负性差碳基
def eg_C(df):
    df['Ele_Gativity_C_1'] = math.pow(1- C_eg   / C_eg, 2) * df['C_mol']  +\
             math.pow(1- Si_eg  / C_eg, 2) * df['Si_mol'] +\
             math.pow(1- Mn_eg  / C_eg, 2) * df['Mn_mol'] +\
             math.pow(1- Ni_eg  / C_eg, 2) * df['Ni_mol'] +\
             math.pow(1- Cr_eg  / C_eg, 2) * df['Cr_mol'] +\
             math.pow(1- Mo_eg  / C_eg, 2) * df['Mo_mol'] +\
             math.pow(1- V_eg   / C_eg, 2) * df['V_mol']  +\
             math.pow(1- Cu_eg  / C_eg, 2) * df['Cu_mol'] +\
             math.pow(1- W_eg   / C_eg, 2) * df['W_mol']  +\
             math.pow(1- Al_eg  / C_eg, 2) * df['Al_mol'] +\
             math.pow(1- Ti_eg  / C_eg, 2) * df['Ti_mol'] +\
             math.pow(1- Nb_eg  / C_eg, 2) * df['Nb_mol'] +\
             math.pow(1- N_eg   / C_eg, 2) * df['N_mol']  +\
             math.pow(1- Co_eg  / C_eg, 2) * df['Co_mol'] +\
             math.pow(1- B_eg   / C_eg, 2) * df['B_mol']  +\
             math.pow(1- P_eg   / C_eg, 2) * df['P_mol']  +\
             math.pow(1- S_eg   / C_eg, 2) * df['S_mol']  +\
             math.pow(1- Fe_eg  / C_eg, 2) * df['Fe_mol']

    df['EN_C'] = df['Ele_Gativity_C_1'].apply(np.sqrt)
    return df

#The effects of alloy elements on lattice constants.
def fcclatpar(df):
    df['FCC_lat'] =0.033*df['C']+ \
           0.0 * df['Si']+ \
           0.00095 * df['Mn']-\
           0.0002 * df['Ni']+\
           0.0006 * df['Cr']+\
           0.0031 * df['Mo']+\
           0.0018  * df['V'] +\
           0.0 * df['Cu']+\
           0.0  * df['W'] +\
           0.0056 * df['Al']+\
           0.0 * df['Ti']+\
           0.0 * df['Nb']+\
           0.0  * df['N'] +\
           0.0 * df['Co']+\
           0.0  * df['B'] +\
           0.0  * df['P'] +\
           0.0  * df['S'] +3.5780
    return df

def bcclatpar(df):
    df['BCC_lat'] =((2.8664-0.279*df['C_mol'])*(2.8664-0.279*df['C_mol'])*(2.8664+2.496*df['C_mol'])-2.8664*2.8664*2.8664)/3/2.8664/2.8664- \
           0.03 * df['Si_mol']+ \
           0.06 * df['Mn_mol']+\
           0.07 * df['Ni_mol']+\
           0.05 * df['Cr_mol']+\
           0.31 * df['Mo_mol']+\
           0.096  * df['V_mol'] +\
           0.0 * df['Cu_mol']+\
           0.0  * df['W_mol'] +\
           0.0 * df['Al_mol']+\
           0.0 * df['Ti_mol']+\
           0.0 * df['Nb_mol']+\
           0.0  * df['N_mol'] +\
           0.0 * df['Co_mol']+\
           0.0  * df['B_mol'] +\
           0.0  * df['P_mol'] +\
           0.0  * df['S_mol']+2.8664
    return df

df_va = ghosh(df_va)

#计算原子百分数
df_va = Atom_fraction(df_va)

df_va = Atom_R(df_va)

df_va = Atom_difference(df_va)

df_va = All_VEC_Cal(df_va)

df_va = VEC_Fe_Cal(df_va)

df_va = VEC_C_Cal(df_va)

df_va = Ele_Gativity(df_va)

df_va = eg_Fe(df_va)

df_va = eg_C(df_va)

df_va = fcclatpar(df_va)

df_va = bcclatpar(df_va)

print(df_va.head())

X_va = df_va[['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S', '∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']]
Y_va = df_va['Ms (K)']
df_va['predict_1'] = reg_ele_fea.predict(X_va)
print('mae:',mean_absolute_error(df_va['Ms (K)'] , df_va['predict_1']))
print('mse:',mean_squared_error(df_va['Ms (K)'] , df_va['predict_1']))
print('rmse:',math.sqrt(mean_squared_error(df_va['Ms (K)'] , df_va['predict_1'])))
print('R2:',r2_score(df_va['Ms (K)'] , df_va['predict_1']))
print('EV:',explained_variance_score(df_va['Ms (K)'] , df_va['predict_1']))



print('\n')
print('mae:',mean_absolute_error(df_va['Ms (K)'] , df_va['cal_Ms']))
print('mse:',mean_squared_error(df_va['Ms (K)'] , df_va['cal_Ms']))
print('rmse:',math.sqrt(mean_squared_error(df_va['Ms (K)'] , df_va['cal_Ms'])))
print('R2:',r2_score(df_va['Ms (K)'] , df_va['cal_Ms']))
print('EV:',explained_variance_score(df_va['Ms (K)'] , df_va['cal_Ms']))


#Summary of domestic and foreign data
#导入数据
benxi = pd.read_csv('./data/chinese_books/benxi.csv')
linhuiguo = pd.read_csv('./data/chinese_books/linhuiguo.csv')
yangkegong = pd.read_csv('./data/chinese_books/yangkegong.csv')
zhangshizhong = pd.read_csv('./data/chinese_books/zhangshizhong.csv')


def feature_eng(df):
    df = ghosh(df)

    df = Atom_fraction(df)

    df = Atom_R(df)

    df = Atom_difference(df)

    df = All_VEC_Cal(df)

    df = VEC_Fe_Cal(df)

    df = VEC_C_Cal(df)

    df = Ele_Gativity(df)

    df = eg_Fe(df)

    df = eg_C(df)

    df = fcclatpar(df)

    df = bcclatpar(df)

    return df

benxi_fea = feature_eng(benxi)
linhuiguo_fea = feature_eng(linhuiguo)
yangkegong_fea = feature_eng(yangkegong)
zhangshizhong_fea = feature_eng(zhangshizhong)

list_1 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S',\
            '∆Gc',\
            'Sum_Atom_R', 'Atom_diff(Fe)',\
            'Sum_VEN',    'VEN_Fe','VEN_C',\
            'Sum_EN','EN_Fe','EN_C',\
            'FCC_lat','BCC_lat','Ms (K)']

benxi_df = benxi_fea[list_1]
yangkegong_df = yangkegong_fea[list_1]
linhuiguo_df = linhuiguo_fea[list_1]
zhangshizhong_df = zhangshizhong_fea[list_1]
data_ch = pd.concat([benxi_df , yangkegong_df , linhuiguo_df , zhangshizhong_df],axis = 0)
list_5 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S', '∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']
label_5 = 'Ms (K)'
reg_ele_fea = LGBMRegressor(n_estimators = 1000,random_state = 123).fit(X_ele_fea  , Y_ele_fea)
X_df_ch = data_ch[list_5]
Y_df_ch = data_ch[label_5]
data_ch['predict_ch'] = reg_ele_fea.predict(X_df_ch)
dif=abs(data_ch['Ms (K)']-data_ch['predict_ch'])
df_clean = data_ch[dif<30]


def bingtu(data, title):
    data.reset_index(inplace=True, drop=True)
    X_da = data[list_5]
    Y_da = data['Ms (K)']
    data['da_pre'] = reg_ele_fea.predict(X_da)

    dif = abs(Y_da - data['da_pre'])

    def cal_acc(a, b):
        n = 0
        for i in range(data.shape[0]):
            if a <= dif[i] < b:
                n = n + 1
        return n

    clo = cal_acc(0, 30)
    app = cal_acc(30, 60)
    far = cal_acc(60, 1000)
    wucha_benxi = pd.Series([clo, app, far], index=['0-30', '30-60', '60<'])
    print(wucha_benxi)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }

    labels = ['0-30', '30-60', '60<']
    explode = (0.05, 0, 0)  # 0.1为第二个元素凸出距离
    colors = ['limegreen', 'deepskyblue', 'tomato']
    plt.figure(figsize=(8, 6))
    plt.pie(wucha_benxi, explode=explode, labels=labels, colors=colors, \
            autopct='%1.2f%%', shadow=False, pctdistance=0.7, \
            startangle=0, textprops={'fontsize': 20, 'color': 'black'})

    plt.title(title, font1)

    plt.legend(loc='upper right')

    path1 = './Figure/3.3/'
    plt.savefig(path1 + title + '.jpg', bbox_inches='tight', dpi=600)

    plt.show()

list_6 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S', '∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat','Ms (K)']
df_5 = df_4[list_6]
df_clean_1 = df_clean[list_6]
data_all_2 = pd.concat([df_clean_1 , df_5],axis = 0)
#根据元素含量，对数据做一个排序
df_all_3 = data_all_2.sort_values(by=['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'], ascending=(True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True))
chongfu = df_all_3.duplicated(subset=['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'])
print(chongfu.shape)
df_all_3.insert(df_all_3.shape[1],'index_chongfu',chongfu)
df_all_3['Ms_mean'] = df_all_3.groupby(['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'])['Ms (K)'].transform('mean')
df_all_3_order = df_all_3.drop_duplicates(subset= ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'], keep='first', inplace=False)

index_2 = np.arange(df_all_3_order.shape[0])
np.random.seed(2021)
np.random.shuffle(index_2)
df_all_3_order.insert(df_all_3_order.shape[1],'index_2',index_2)
df_all_4 = df_all_3_order.sort_values(by=['index_2'], ascending=(True))
df_all_4.set_index('index_2',inplace=True)
df_all_4.reset_index(inplace = True)

X_all = df_all_4[['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S', '∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']]
Y_all = df_all_4['Ms_mean']
X_train_all , X_test_all , y_train_all , y_test_all = TTS(X_all , Y_all , test_size=0.1, random_state=5)
reg_all = LGBMRegressor(
    n_estimators = 1000,
    random_state = 123).fit(X_train_all  , y_train_all )
df_va['predict_2'] = reg_all.predict(X_va)

print('mae:',mean_absolute_error(df_va['Ms (K)'] , df_va['predict_2']))
print('mse:',mean_squared_error(df_va['Ms (K)'] , df_va['predict_2']))
print('rmse:',math.sqrt(mean_squared_error(df_va['Ms (K)'] , df_va['predict_2'])))

print('R2:',r2_score(df_va['Ms (K)'] , df_va['predict_2']))
print('EV:',explained_variance_score(df_va['Ms (K)'] , df_va['predict_2']))