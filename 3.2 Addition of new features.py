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

df_3 = pd.read_csv('./data/df_3.csv')
def calculation_canshu(df,list_1,label):
    X_ele = df[list_1]
    Y_ele = df[label]
    X_train , X_test , y_train , y_test = TTS(X_ele , Y_ele , test_size=0.1, random_state=5)
    reg = LGBMRegressor(n_estimators = 1000,random_state = 123).fit(X_train  , y_train)
    y_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test , y_reg)
    mse = mean_squared_error(y_test , y_reg)
    rmse = math.sqrt(mean_squared_error(y_test , y_reg))
    r2 = r2_score(y_test , y_reg)
    ev = explained_variance_score(y_test , y_reg)
    return mae, mse, rmse, r2, ev

#Addition of thermodynamic driving force values

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


df_4 = ghosh(df_3)

df_4 = Atom_fraction(df_4)

df_4 = Atom_R(df_4)

df_4 = Atom_difference(df_4)

df_4 = All_VEC_Cal(df_4)

df_4 = VEC_Fe_Cal(df_4)

df_4 = VEC_C_Cal(df_4)

df_4 = Ele_Gativity(df_4)

df_4 = eg_Fe(df_4)

df_4 = eg_C(df_4)

df_4 = fcclatpar(df_4)

df_4 = bcclatpar(df_4)

list_no = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']
label_no = 'Ms (K)'
mae_no, mse_no, rmse_no, r2_no, ev_no = calculation_canshu(df_4,list_no, label_no)

list_ghosh = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','∆Gc']
label_ghosh = 'Ms (K)'
mae_ghosh, mse_ghosh, rmse_ghosh, r2_ghosh, ev_ghosh = calculation_canshu(df_4,list_ghosh, label_ghosh)

list_Fe_mol  = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','Fe_mol']
label_Fe_mol = 'Ms (K)'
mae_Fe_mol, mse_Fe_mol, rmse_Fe_mol, r2_Fe_mol, ev_Fe_mol = calculation_canshu(df_4, list_Fe_mol, label_Fe_mol)

list_Atom_R = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','Sum_Atom_R']
label_Atom_R = 'Ms (K)'
mae_Atom_R, mse_Atom_R, rmse_Atom_R, r2_Atom_R, ev_Atom_R = calculation_canshu(df_4, list_Atom_R, label_Atom_R)

list_Atom_diff_Fe = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','Atom_diff(Fe)']
label_Atom_diff_Fe= 'Ms (K)'
mae_Atom_diff_Fe, mse_Atom_diff_Fe, rmse_Atom_diff_Fe, r2_Atom_diff_Fe, ev_Atom_diff_Fe = calculation_canshu(df_4, list_Atom_diff_Fe, label_Atom_diff_Fe)

list_Sum_VEC = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','Sum_VEN']
label_Sum_VEC = 'Ms (K)'
mae_Sum_VEC, mse_Sum_VEC, rmse_Sum_VEC, r2_Sum_VEC, ev_Sum_VEC = calculation_canshu(df_4, list_Sum_VEC, label_Sum_VEC)

list_VEC_Fe  = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','VEN_Fe']
label_VEC_Fe = 'Ms (K)'
mae_VEC_Fe, mse_VEC_Fe, rmse_VEC_Fe, r2_VEC_Fe, ev_VEC_Fe = calculation_canshu(df_4, list_VEC_Fe, label_VEC_Fe)

list_VEC_C = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','VEN_C']
label_VEC_C = 'Ms (K)'
mae_VEC_C, mse_VEC_C, rmse_VEC_C, r2_VEC_C, ev_VEC_C = calculation_canshu(df_4, list_VEC_C, label_VEC_C)

list_Sum_Ele_Gativity = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','Sum_EN']
label_Sum_Ele_Gativity = 'Ms (K)'
mae_Sum_Ele_Gativity, mse_Sum_Ele_Gativity, rmse_Sum_Ele_Gativity, r2_Sum_Ele_Gativity, ev_Sum_Ele_Gativity= calculation_canshu(df_4, list_Sum_Ele_Gativity, label_Sum_Ele_Gativity)

list_Ele_Gativity_Fe = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','EN_Fe']
label_Ele_Gativity_Fe = 'Ms (K)'
mae_Fe_Ele_Gativity, mse_Fe_Ele_Gativity, rmse_Fe_Ele_Gativity, r2_Fe_Ele_Gativity, ev_Fe_Ele_Gativity = calculation_canshu(df_4, list_Ele_Gativity_Fe, label_Ele_Gativity_Fe)

list_Ele_Gativity_C  = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','EN_C']
label_Ele_Gativity_C = 'Ms (K)'
mae_C_Ele_Gativity, mse_C_Ele_Gativity, rmse_C_Ele_Gativity, r2_C_Ele_Gativity, ev_C_Ele_Gativity = calculation_canshu(df_4, list_Ele_Gativity_C, label_Ele_Gativity_C)

list_fcc  = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','FCC_lat']
label_fcc = 'Ms (K)'
mae_fcc, mse_fcc, rmse_fcc, r2_fcc, ev_fcc = calculation_canshu(df_4,list_fcc, label_fcc)

list_bcc  = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S','BCC_lat']
label_bcc = 'Ms (K)'
mae_bcc, mse_bcc, rmse_bcc, r2_bcc, ev_bcc = calculation_canshu(df_4,list_bcc, label_bcc)

list_all = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S',\
            '∆Gc',\
            'Sum_Atom_R', 'Atom_diff(Fe)',\
            'Sum_VEN',    'VEN_Fe','VEN_C',\
            'Sum_EN','EN_Fe','EN_C',\
            'FCC_lat','BCC_lat']
label_all = 'Ms (K)'
mae_all, mse_all, rmse_all, r2_all, ev_all = calculation_canshu(df_4,list_all, label_all)

list_5 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S', '∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']
label_5 = 'Ms (K)'
mae_5, mse_5, rmse_5, r2_5, ev_5 = calculation_canshu(df_4, list_5, label_5)

import matplotlib.pyplot as plt
import numpy as np
x = ['Just Element', '∆Gc', 'Sum_Atom_R', 'Atom_diff(Fe)', 'Sum_VEN', \
     'VEN_Fe', 'VEN_C', 'Sum_EN', 'EN_Fe', 'EN_C','FCC_lat','BCC_lat', 'All_Feature','Better_feature']
y = [mae_no, mae_ghosh,  mae_Atom_R, mae_Atom_diff_Fe, mae_Sum_VEC, mae_VEC_Fe, mae_VEC_C, mae_Sum_Ele_Gativity, mae_Fe_Ele_Gativity, \
     mae_C_Ele_Gativity, mae_fcc, mae_bcc , mae_all , mae_5]
plt.figure(figsize=(10,5))
plt.tick_params(labelsize=15)
plt.xticks(rotation=60)
plt.grid(ls='--')
plt.plot(x, y)

beneficial_x = ['∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']
beneficial_y = [mae_ghosh, mae_C_Ele_Gativity, mae_Sum_VEC, mae_VEC_C, mae_fcc]

plt.scatter(beneficial_x , beneficial_y , s=80 , color='red' , alpha=0.8 , marker='o')
plt.scatter(x[-1],y[-1] , s=150 , color='g' , alpha=0.8 , marker='o')
plt.scatter(x[0],y[0] , s=150 , color='b' , alpha=0.8 , marker='o')

plt.title('Error value when adding different features(MAE)',fontsize = 20)
plt.axhline(mae_no, color='r', linestyle='--', label='xxx')
plt.axhline(mae_5, color='g', linestyle='--', label='xxx')
plt.savefig("./Figure/3.2/MAE_Feature.JPEG" , dpi=600, bbox_inches = 'tight')
plt.show()



x = ['Just Element', '∆Gc', 'Sum_Atom_R', 'Atom_diff(Fe)', 'Sum_VEN', \
     'VEN_Fe', 'VEN_C', 'Sum_EN', 'EN_Fe', 'EN_C','FCC_lat','BCC_lat', 'All_Feature','Better_feature']
y = [rmse_no, rmse_ghosh,  rmse_Atom_R, rmse_Atom_diff_Fe, rmse_Sum_VEC, rmse_VEC_Fe, rmse_VEC_C, rmse_Sum_Ele_Gativity, rmse_Fe_Ele_Gativity, \
     rmse_C_Ele_Gativity, rmse_fcc, rmse_bcc , rmse_all , rmse_5]
plt.figure(figsize=(10,5))
plt.tick_params(labelsize=15)
plt.xticks(rotation=60)
plt.grid(ls='--')
plt.plot(x, y)

beneficial_x = ['∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']
beneficial_y = [rmse_ghosh, rmse_C_Ele_Gativity, rmse_Sum_VEC, rmse_VEC_C, rmse_fcc]
plt.scatter(beneficial_x , beneficial_y , s=80 , color='red' , alpha=0.8 , marker='o')
plt.scatter(x[-1],y[-1] , s=150 , color='g' , alpha=0.8 , marker='o')
plt.scatter(x[0],y[0] , s=150 , color='b' , alpha=0.8 , marker='o')

plt.title('Error value when adding different features(RMSE)',fontsize = 20)
plt.axhline(rmse_no, color='r', linestyle='--', label='xxx')
plt.axhline(rmse_5, color='g', linestyle='--', label='xxx')
plt.savefig("./Figure/3.2/RMSE_Feature.JPEG" , dpi=600, bbox_inches = 'tight')
plt.show()

x = ['Just Element', '∆Gc', 'Sum_Atom_R', 'Atom_diff(Fe)', 'Sum_VEN', \
     'VEN_Fe', 'VEN_C', 'Sum_EN', 'EN_Fe', 'EN_C','FCC_lat','BCC_lat', 'All_Feature','Better_feature']
y = [r2_no, r2_ghosh,  r2_Atom_R, r2_Atom_diff_Fe, r2_Sum_VEC, r2_VEC_Fe, r2_VEC_C, r2_Sum_Ele_Gativity, r2_Fe_Ele_Gativity, \
     r2_C_Ele_Gativity, r2_fcc, r2_bcc , r2_all , r2_5]
plt.figure(figsize=(10,5))
plt.tick_params(labelsize=15)
plt.xticks(rotation=60)  #修改x轴刻度，并将刻度旋转30度
plt.grid(ls='--')  # 生成网格
plt.plot(x, y)

beneficial_x = ['∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']
beneficial_y = [r2_ghosh, r2_C_Ele_Gativity, r2_Sum_VEC, r2_VEC_C, r2_fcc]
#传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
plt.scatter(beneficial_x , beneficial_y , s=80 , color='red' , alpha=0.8 , marker='o')
plt.scatter(x[-1],y[-1] , s=150 , color='g' , alpha=0.8 , marker='o')
plt.scatter(x[0],y[0] , s=150 , color='b' , alpha=0.8 , marker='o')

plt.title('Error value when adding different features(R2)',fontsize = 20)
plt.axhline(r2_no, color='r', linestyle='--', label='xxx')
plt.axhline(r2_5, color='g', linestyle='--', label='xxx')
plt.savefig("./Figure/3.2/R2_Feature.JPEG" , dpi=600, bbox_inches = 'tight')
plt.show()

x = ['Just Element', '∆Gc', 'Sum_Atom_R', 'Atom_diff(Fe)', 'Sum_VEN', \
     'VEN_Fe', 'VEN_C', 'Sum_EN', 'EN_Fe', 'EN_C','FCC_lat','BCC_lat', 'All_Feature','Better_feature']
y = [ev_no, ev_ghosh,  ev_Atom_R, ev_Atom_diff_Fe, ev_Sum_VEC, ev_VEC_Fe, ev_VEC_C, ev_Sum_Ele_Gativity, ev_Fe_Ele_Gativity, \
     ev_C_Ele_Gativity, ev_fcc, ev_bcc , ev_all , ev_5]
plt.figure(figsize=(10,5))
plt.tick_params(labelsize=15)
plt.xticks(rotation=60)  #修改x轴刻度，并将刻度旋转30度
plt.grid(ls='--')  # 生成网格
plt.plot(x, y)

beneficial_x = ['∆Gc','EN_C','Sum_VEN','VEN_C','FCC_lat']
beneficial_y = [ev_ghosh, ev_C_Ele_Gativity, ev_Sum_VEC, ev_VEC_C, ev_fcc]
#传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
plt.scatter(beneficial_x , beneficial_y , s=80 , color='red' , alpha=0.8 , marker='o')
plt.scatter(x[-1],y[-1] , s=150 , color='g' , alpha=0.8 , marker='o')
plt.scatter(x[0],y[0] , s=150 , color='b' , alpha=0.8 , marker='o')


plt.title('Error value when adding different features(EV)',fontsize = 20)
plt.axhline(ev_no, color='r', linestyle='--', label='xxx')
plt.axhline(ev_5, color='g', linestyle='--', label='xxx')
plt.savefig("./Figure/3.2/EV_Feature.JPEG" , dpi=600, bbox_inches = 'tight')
plt.show()

df_4.to_csv('./data/df_4.csv')