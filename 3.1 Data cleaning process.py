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

#Calculation function
def calculate_re(df, list_1, label, calculate_clean):
    X_ele_1 = df[list_1]
    Y_ele_1 = df[label]
    X_train_ele, X_test_ele, y_train_ele, y_test_ele = TTS(X_ele_1, Y_ele_1, test_size=0.1, random_state=5)
    reg_ele_1 = LGBMRegressor(n_estimators=1000, random_state=123).fit(X_train_ele, y_train_ele)
    y_reg_ele = reg_ele_1.predict(X_test_ele)

    print("MAE_Error：", mean_absolute_error(y_test_ele, y_reg_ele))
    MAE = mean_absolute_error(y_test_ele, y_reg_ele)

    print("MSE_Error：", mean_squared_error(y_test_ele, y_reg_ele))
    MSE = mean_squared_error(y_test_ele, y_reg_ele)

    print("RMSE_Error：", math.sqrt(mean_squared_error(y_test_ele, y_reg_ele)))
    RMSE = math.sqrt(mean_squared_error(y_test_ele, y_reg_ele))

    print("R2_Error：", r2_score(y_test_ele, y_reg_ele))
    R2 = r2_score(y_test_ele, y_reg_ele)

    print("EV_Error：", explained_variance_score(y_test_ele, y_reg_ele))
    EV = explained_variance_score(y_test_ele, y_reg_ele)
    return MAE, MSE, RMSE, R2, EV
#Data acquisition
#Import raw data
df_1 = pd.read_csv('./data/MAP_data/Original_MAP_DATA.csv')
print(df_1.head())

#Missing and duplicate data processing

df_1_sorted = df_1.sort_values(by=['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', \
                                   'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'], \
                               ascending=(True, True, True, True, True, True, True, \
                                          True, True, True, True, True, True, True, True, True, True))

Duplicate_data = df_1_sorted.duplicated(subset=['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', \
                                         'Nb', 'Ti', 'B', 'N', 'P', 'S'],keep= 'first')
print('Number of repeated data items：',Duplicate_data.sum())
df_1_sorted.insert(df_1_sorted.shape[1],'index_duplicate',Duplicate_data)
df_1_sorted['Ms_mean'] = df_1_sorted.groupby(['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co',\
                                              'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'])['Ms (K)'].transform('mean')
df_1_sorted.drop_duplicates(subset= ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', \
                                     'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'], keep='first', inplace=True)
df_2 = df_1_sorted
#Data sorting
index_1 = np.arange(df_2.shape[0])
np.random.seed(123)
np.random.shuffle(index_1)
df_2['index_1'] = index_1
df_2.sort_values(by=['index_1'], ascending=(True),inplace=True)
df_2.set_index('index_1',inplace=True)
df_2.reset_index(inplace=True)

df_2.drop(['Ms (K)', 'index_duplicate','index_1'],axis=1,inplace=True)
df_2.rename(columns={"Ms_mean": "Ms (K)"},inplace=True)
ele_list_2 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']
label_2 = 'Ms (K)'
calculate_re(df_2, ele_list_2, label_2,1)

#Data visualization analysis
feature_list = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W',\
                'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S','Ms (K)']

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
n = 1
plt.figure(figsize=(14, 14), dpi=100)
plt.tight_layout(h_pad=10)
for i in feature_list:
    plt.subplot(5, 4, n)
    plt.tight_layout(h_pad=1)
    plt.hist(df_2[i], edgecolor="black", bins=50)
    plt.tick_params(labelsize=15)
    plt.title(i, font)  # 标题，并设定字号大小
    # plt.xlabel(name[n-1],fontdict = font)
    # ax.tick_params(labelsize=20)

    n = n + 1
plt.savefig('./Figure/3.1/data_distribution.png')

ele = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S']
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 25,
}

df = pd.DataFrame(df_2[ele] , columns=ele)
plt.figure(figsize=(20,10))
f = df.boxplot(sym = 'o',
               vert = False,
               fontsize = 24,
               whis=1.5,
               patch_artist = True,
               meanline = False,showmeans = True,
               showbox = True,
               showfliers = True,
               notch = False,
               return_type='dict')
plt.title('Box plot of element content',font2)
plt.scatter(10.18, 2 , s=1000 , color='', alpha = 1 , marker='o', edgecolors='r')
plt.scatter(30.00, 8 , s=1000 , color='', alpha = 1 , marker='o', edgecolors='r')
plt.savefig("./Figure/3.1/Box-plot-of-element-content.png")

plt.show()

#Eliminated duplicated entries
df_2 = df_2[df_2['Mn']<10]
df_2 = df_2[df_2['C']>0]
df_2 = df_2[df_2['Co']<20]

df_2.eval('Payson_Savage = 273.15 + 498.9 - 316.7*C - 33.3*Mn - 27.8*Cr - 16.7*Ni - 11.1*(Si +Mo +W)' , inplace=True)
df_2.eval('Andrews_2 = 273.15 + 512 - 453*C - 16.9*Ni + 15*Cr - 9.5*Mo +217*C*C -71.5*Mn*C  - 67.6*Cr*C' , inplace=True)
df_2.eval('Capdevila = 764.2 - 302.6*C - 30.6*Mn -14.5*Si - 8.9*Cr - 16.6*Ni + 2.4*Mo + 8.58*Co - 11.3*Cu + 7.4*W' , inplace=True)

ele_list_2 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']
label_2 = 'Ms (K)'
print('\n')
print('Error of preliminary data cleaning:')
calculate_re(df_2, ele_list_2, label_2,1)

#Deeper data cleaning with reasonable strategy
df_2.reset_index(inplace=True)


def data_clean(df):
    X_1 = df[['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'V', 'Cu', 'W', 'Al', 'Ti', 'Nb', 'N', 'Co', 'B', 'P', 'S']]
    Y_1 = df['Ms (K)']
    reg_1 = LGBMRegressor(n_estimators=1000, random_state=123)
    predicted_1 = cross_val_predict(reg_1, X_1, Y_1, cv=10)
    df['pre_1'] = predicted_1

    r1 = np.zeros((5, 7), dtype=np.double)
    k = 0
    df_10 = pd.DataFrame()
    for m in [10, 20, 30, 40, 50]:
        index_clean = np.zeros((df.shape[0],), dtype=np.int)
        Ms_thr = m
        for i in range(df.shape[0]):
            if abs(df['pre_1'][i] - df['Ms (K)'][i]) > Ms_thr and \
                    abs(df['Payson_Savage'][i] - df['Ms (K)'][i]) > Ms_thr and \
                    abs(df['Andrews_2'][i] - df['Ms (K)'][i]) > Ms_thr and \
                    abs(df['Capdevila'][i] - df['Ms (K)'][i]) > Ms_thr:
                index_clean[i] = 1
            # Ms点过大的条目数量
        sum_1 = index_clean.sum()
        df['index_clean'] = index_clean
        df_cle = df[df['index_clean'] == 0]

        X_cle = df_cle[['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'V', 'Cu', 'W', 'Al', 'Ti', 'Nb', 'N', 'Co', 'B', 'P', 'S']]
        Y_cle = df_cle['Ms (K)']
        X_train_cle, X_test_cle, y_train_cle, y_test_cle = TTS(X_cle, Y_cle, test_size=0.1, random_state=5)

        reg_cle = LGBMRegressor(n_estimators=1000, random_state=123).fit(X_train_cle, y_train_cle)

        y_pred_cle = reg_cle.predict(X_test_cle)

        r1[k][0] = mean_absolute_error(y_test_cle, y_pred_cle)
        r1[k][1] = mean_squared_error(y_test_cle, y_pred_cle)
        r1[k][2] = math.sqrt(mean_squared_error(y_test_cle, y_pred_cle))
        r1[k][3] = r2_score(y_test_cle, y_pred_cle)
        r1[k][4] = explained_variance_score(y_test_cle, y_pred_cle)
        r1[k][5] = sum_1
        r1[k][6] = m

        colname_1 = '{}_rmse'.format(m)
        colname_2 = '{}_mae'.format(m)
        colname_3 = '{}_r2'.format(m)
        colname_4 = '{}_ev'.format(m)

        df_10[colname_1] = cross_val_score(reg_cle, X_cle, Y_cle, verbose=0, scoring='neg_root_mean_squared_error',
                                           cv=10)

        df_10[colname_2] = cross_val_score(reg_cle, X_cle, Y_cle, verbose=0, scoring='neg_median_absolute_error', cv=10)

        df_10[colname_3] = cross_val_score(reg_cle, X_cle, Y_cle, verbose=0, scoring='r2', cv=10)

        df_10[colname_4] = cross_val_score(reg_cle, X_cle, Y_cle, verbose=0, scoring='explained_variance', cv=10)

        k = k + 1
    return r1, df_10

r1,df_10 = data_clean(df_2)
df_test = pd.DataFrame(r1,columns = ['MAE', 'MSE', 'RMSE', 'R2', 'EV', 'shuliang', 'yuzhi'])
print(df_test.head(5))
print(df_10.T)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

tick_label = ['MAE', 'RMSE']

plt.figure(figsize=(12, 8))
plt.title('MAE-RMSE', font1)  # 标题，并设定字号大小

barWidth = 0.15

# 设置柱子的高度
bars1 = [df_test['MAE'][0], df_test['RMSE'][0]]
bars2 = [df_test['MAE'][1], df_test['RMSE'][1]]
bars3 = [df_test['MAE'][2], df_test['RMSE'][2]]
bars4 = [df_test['MAE'][3], df_test['RMSE'][3]]
bars5 = [df_test['MAE'][4], df_test['RMSE'][4]]

r1 = np.arange(len(bars1))
r2 = [x + barWidth * 1.2 for x in r1]
r3 = [x + barWidth * 1.2 for x in r2]
r4 = [x + barWidth * 1.2 for x in r3]
r5 = [x + barWidth * 1.2 for x in r4]

# 创建柱子
plt.bar(r1, bars1, width=barWidth, alpha=0.6, facecolor='cyan', edgecolor='cyan', lw=1, label='Threshold value(10)')
plt.bar(r2, bars2, width=barWidth, alpha=1, facecolor='r', edgecolor='r', lw=1, label='Threshold value(20)')
plt.bar(r3, bars3, width=barWidth, alpha=1, facecolor='lime', edgecolor='lime', lw=1, label='Threshold value(30)')
plt.bar(r4, bars4, width=barWidth, alpha=1, facecolor='darkblue', edgecolor='darkblue', lw=1,
        label='Threshold value(40)')
plt.bar(r5, bars5, width=barWidth, alpha=1, facecolor='darkblue', edgecolor='darkblue', lw=1,
        label='Threshold value(50)')

for x, y in enumerate(bars1):
    plt.text(x, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars2):
    plt.text(x + 0.17, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars3):
    plt.text(x + 0.36, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars4):
    plt.text(x + 0.54, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars5):
    plt.text(x + 0.72, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

# 添加x轴名称
plt.xticks([r + 1.8 * barWidth for r in range(len(bars1))], tick_label)
plt.tick_params(labelsize=15)
# 创建图例
plt.legend(prop=font1, loc=2)
# 展示图片
plt.savefig("./Figure/3.1/deeper-cleaning-ten-fold-MAE-RMSE.JPEG", dpi=600, bbox_inches='tight')
plt.show()

tick_label = ['R2', 'EV']

plt.figure(figsize=(12, 8))
plt.title('R2-EV', font1)  # 标题，并设定字号大小

barWidth = 0.15

# 设置柱子的高度
bars1 = [df_test['R2'][0], df_test['EV'][0]]
bars2 = [df_test['R2'][1], df_test['EV'][1]]
bars3 = [df_test['R2'][2], df_test['EV'][2]]
bars4 = [df_test['R2'][3], df_test['EV'][3]]
bars5 = [df_test['R2'][4], df_test['EV'][4]]

r1 = np.arange(len(bars1))
r2 = [x + barWidth * 1.2 for x in r1]
r3 = [x + barWidth * 1.2 for x in r2]
r4 = [x + barWidth * 1.2 for x in r3]
r5 = [x + barWidth * 1.2 for x in r4]

# 创建柱子
plt.bar(r1, bars1, width=barWidth, alpha=0.6, facecolor='cyan', edgecolor='cyan', lw=1, label='Threshold value(10)')
plt.bar(r2, bars2, width=barWidth, alpha=1, facecolor='r', edgecolor='r', lw=1, label='Threshold value(20)')
plt.bar(r3, bars3, width=barWidth, alpha=1, facecolor='lime', edgecolor='lime', lw=1, label='Threshold value(30)')
plt.bar(r4, bars4, width=barWidth, alpha=1, facecolor='darkblue', edgecolor='darkblue', lw=1,
        label='Threshold value(40)')
plt.bar(r5, bars5, width=barWidth, alpha=1, facecolor='darkblue', edgecolor='darkblue', lw=1,
        label='Threshold value(50)')

for x, y in enumerate(bars1):
    plt.text(x, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars2):
    plt.text(x + 0.17, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars3):
    plt.text(x + 0.36, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars4):
    plt.text(x + 0.54, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars5):
    plt.text(x + 0.72, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

# 添加x轴名称
plt.xticks([r + 1.8 * barWidth for r in range(len(bars1))], tick_label)
plt.tick_params(labelsize=15)
# 创建图例
# plt.legend(prop=font1,loc=2)
# 展示图片
plt.savefig("./Figure/3.1/deeper-cleaning-ten-fold--R2-R2.JPEG", dpi=600, bbox_inches='tight')
plt.show()



rmse_10_mean = abs(df_10['10_rmse'].mean())
rmse_20_mean = abs(df_10['20_rmse'].mean())
rmse_30_mean = abs(df_10['30_rmse'].mean())
rmse_40_mean = abs(df_10['40_rmse'].mean())
rmse_50_mean = abs(df_10['50_rmse'].mean())
rmse_10_max = abs(df_10['10_rmse'].min())
rmse_20_max = abs(df_10['20_rmse'].min())
rmse_30_max = abs(df_10['30_rmse'].min())
rmse_40_max = abs(df_10['40_rmse'].min())
rmse_50_max = abs(df_10['50_rmse'].min())

tick_label = ['Mean', 'Max']

plt.figure(figsize=(12, 8))
plt.title('The calculation results(RMSE) in the step3', font1)  # 标题，并设定字号大小

barWidth = 0.15

# 设置柱子的高度
bars1 = [rmse_10_mean, rmse_10_max]
bars2 = [rmse_20_mean, rmse_20_max]
bars3 = [rmse_30_mean, rmse_30_max]
bars4 = [rmse_40_mean, rmse_40_max]
bars5 = [rmse_50_mean, rmse_50_max]

r1 = np.arange(len(bars1))
r2 = [x + barWidth * 1.2 for x in r1]
r3 = [x + barWidth * 1.2 for x in r2]
r4 = [x + barWidth * 1.2 for x in r3]
r5 = [x + barWidth * 1.2 for x in r4]

# 创建柱子
plt.bar(r1, bars1, width=barWidth, alpha=0.6, facecolor='blue', edgecolor='blue', lw=1, label='Threshold value(10)')
plt.bar(r2, bars2, width=barWidth, alpha=1, facecolor='orange', edgecolor='orange', lw=1, label='Threshold value(20)')
plt.bar(r3, bars3, width=barWidth, alpha=1, facecolor='green', edgecolor='green', lw=1, label='Threshold value(30)')
plt.bar(r4, bars4, width=barWidth, alpha=1, facecolor='red', edgecolor='red', lw=1, label='Threshold value(40)')
plt.bar(r5, bars5, width=barWidth, alpha=1, facecolor='purple', edgecolor='purple', lw=1, label='Threshold value(50)')

for x, y in enumerate(bars1):
    plt.text(x, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars2):
    plt.text(x + 0.17, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars3):
    plt.text(x + 0.36, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars4):
    plt.text(x + 0.54, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars5):
    plt.text(x + 0.72, y + 0.005, '%s' % round(y, 2), ha='center', va='bottom', fontsize=15)

# 添加x轴名称
plt.xticks([r + 2.4 * barWidth for r in range(len(bars1))], tick_label)
plt.tick_params(labelsize=15)
# 创建图例
plt.legend(prop=font1, loc=2)
# 展示图片


plt.savefig("./Figure/3.1/deeper-cleaning-RMSE_minmax.JPEG", dpi=600, bbox_inches='tight')
plt.show()




r2_10_mean = df_10['10_r2'].mean()
r2_20_mean = df_10['20_r2'].mean()
r2_30_mean = df_10['30_r2'].mean()
r2_40_mean = df_10['40_r2'].mean()
r2_50_mean = df_10['50_r2'].mean()

r2_10_min = df_10['10_r2'].min()
r2_20_min = df_10['20_r2'].min()
r2_30_min = df_10['30_r2'].min()
r2_40_min = df_10['40_r2'].min()
r2_50_min = df_10['50_r2'].min()

tick_label = ['Mean', 'Min']

plt.figure(figsize=(12, 8))
plt.title('The calculation results(R2) in the step3', font1)  # 标题，并设定字号大小

barWidth = 0.15

# 设置柱子的高度
bars1 = [r2_10_mean, r2_10_min]
bars2 = [r2_20_mean, r2_20_min]
bars3 = [r2_30_mean, r2_30_min]
bars4 = [r2_40_mean, r2_40_min]
bars5 = [r2_50_mean, r2_50_min]

r1 = np.arange(len(bars1))
r2 = [x + barWidth * 1.2 for x in r1]
r3 = [x + barWidth * 1.2 for x in r2]
r4 = [x + barWidth * 1.2 for x in r3]
r5 = [x + barWidth * 1.2 for x in r4]

# 创建柱子
plt.bar(r1, bars1, width=barWidth, alpha=0.6, facecolor='blue', edgecolor='blue', lw=1, label='step_1')
plt.bar(r2, bars2, width=barWidth, alpha=1, facecolor='orange', edgecolor='orange', lw=1, label='step_2')
plt.bar(r3, bars3, width=barWidth, alpha=1, facecolor='green', edgecolor='green', lw=1, label='step_3')
plt.bar(r4, bars4, width=barWidth, alpha=1, facecolor='red', edgecolor='red', lw=1, label='step_4')
plt.bar(r5, bars5, width=barWidth, alpha=1, facecolor='purple', edgecolor='purple', lw=1, label='step_5')

for x, y in enumerate(bars1):
    plt.text(x, y + 0.005, '%s' % round(y, 3), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars2):
    plt.text(x + 0.17, y + 0.005, '%s' % round(y, 3), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars3):
    plt.text(x + 0.36, y + 0.005, '%s' % round(y, 3), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars4):
    plt.text(x + 0.54, y + 0.005, '%s' % round(y, 3), ha='center', va='bottom', fontsize=15)

for x, y in enumerate(bars5):
    plt.text(x + 0.72, y + 0.005, '%s' % round(y, 3), ha='center', va='bottom', fontsize=15)

# 添加x轴名称
plt.xticks([r + 2.4 * barWidth for r in range(len(bars1))], tick_label)
plt.tick_params(labelsize=15)
# 创建图例
# plt.legend(prop=font1,loc=2)
# 展示图片


plt.savefig("./Figure/3.1/deeper-cleaning-R2_min_max.JPEG", dpi=600, bbox_inches='tight')
plt.show()


X_2 = df_2[['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']]
Y_2 = df_2['Ms (K)']
reg_2 = LGBMRegressor(n_estimators = 1000,random_state = 123)
predicted_3 = cross_val_predict(reg_2 , X_2 , Y_2 , cv=10)
df_2['pre_3'] = predicted_3


index_clean = np.zeros((df_2.shape[0],), dtype=np.int)
Ms_thr = 30
for i in range(df_2.shape[0]):
    if abs(df_2['pre_3'][i] - df_2['Ms (K)'][i])>Ms_thr and abs(df_2['Payson_Savage'][i] - df_2['Ms (K)'][i])>Ms_thr and \
       abs(df_2['Andrews_2'][i] - df_2['Ms (K)'][i])>Ms_thr and \
       abs(df_2['Capdevila'][i] - df_2['Ms (K)'][i])>Ms_thr:
        index_clean[i]=1
sum_1 = index_clean.sum()
df_2['index_clean'] = index_clean
df_2_new = df_2[df_2['index_clean']==0]

df_3 = df_2_new
X_ele_3 = df_3[['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']]
Y_ele_3 = df_3['Ms (K)']

reg_ele_3 = LGBMRegressor(n_estimators = 1000, random_state = 123)

predicted_3 = cross_val_predict(reg_ele_3 , X_ele_3 , Y_ele_3 , cv=10)
df_3['predicted_3'] = predicted_3

ele_list_3 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']
label_3 = 'Ms (K)'
print('\n')
print('Error of preliminary data cleaning:')
calculate_re(df_3, ele_list_3, label_3,1)
df_3.to_csv('./data/df_3.csv')