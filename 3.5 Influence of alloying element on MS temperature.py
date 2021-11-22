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

df_1 = pd.read_csv('./data/MAP_data/Original_MAP_DATA.csv')
#Missing and duplicate data processing
df_1_sorted = df_1.sort_values(by=['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', \
                                   'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'], \
                               ascending=(True, True, True, True, True, True, True, \
                                          True, True, True, True, True, True, True, True, True, True))
#计算重复数据的数量
chongfu = df_1_sorted.duplicated(subset=['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', \
                                         'Nb', 'Ti', 'B', 'N', 'P', 'S'],keep= 'first')
print('重复数据条数：',chongfu.sum())
df_1_sorted.insert(df_1_sorted.shape[1],'index_chongfu',chongfu)

#对于元素含量重复，但Ms点不一致的数据，取其Ms点平均值

df_1_sorted['Ms_mean'] = df_1_sorted.groupby(['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co',\
                                              'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'])['Ms (K)'].transform('mean')

#清洗掉重复数据
df_2 = df_1_sorted.drop_duplicates(subset= ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', \
                                     'W', 'Cu', 'Nb', 'Ti', 'B', 'N', 'P', 'S'], keep='first', inplace=True)
df_2 = df_1_sorted
index_1 = np.arange(df_2.shape[0])
np.random.seed(123)
np.random.shuffle(index_1)
df_2['index_1'] = index_1
df_2.sort_values(by=['index_1'], ascending=(True),inplace=True)
df_2.set_index('index_1',inplace=True)
df_2.reset_index(inplace=True)

df_2.drop(['Ms (K)', 'index_chongfu','index_1'],axis=1,inplace=True)
df_2.rename(columns={"Ms_mean": "Ms (K)"},inplace=True)
ele_list_2 = ['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']
label_2 = 'Ms (K)'
calculate_re(df_2, ele_list_2, label_2,1)

df_2 = df_2[df_2['Mn']<10]
df_2 = df_2[df_2['C']>0]
df_2 = df_2[df_2['Co']<20]

X_ele = df_2[['C','Si','Mn','Ni','Cr','Mo','V','Cu','W','Al','Ti','Nb','N','Co','B','P','S']]
Y_ele = df_2['Ms (K)']

reg_ele = LGBMRegressor(n_estimators = 1000,random_state = 123).fit(X_ele  , Y_ele)

font_yz = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
lgb.plot_importance(reg_ele , ax=None , color='r' , height=0.2 ,\
                    xlim=None , ylim=None , title='Feature importance(gain)' ,\
                    xlabel='Feature importance' , ylabel='Features' , \
                    importance_type = 'gain' , max_num_features = None, \
                    ignore_zero=True, figsize=(12,9), dpi = 100, grid=True, precision=1)
plt.title("Feature importance(gain)",fontdict = font_yz)
plt.yticks(fontsize=15)
plt.xlabel('Feature importance',fontdict = font_yz)
plt.ylabel('Features',fontdict = font_yz)

plt.savefig('./Figure/3.5/Feature_importance(gain)_1.png', dpi = 600,bbox_inches = 'tight')



lgb.plot_importance(reg_ele, ax=None, height=0.2, color='lime', \
                    xlim=None, ylim=None, title='Feature importance(split)',\
                    xlabel='Feature importance', ylabel='Features', importance_type='split', \
                    max_num_features=None, ignore_zero=True, figsize=(12,9), dpi = 100, grid=True, precision=1)

plt.title("Feature importance(split)",fontdict = font_yz)
plt.yticks(fontsize=15)
plt.xlabel('Feature importance',fontdict = font_yz)
plt.ylabel('Features',fontdict = font_yz)
plt.savefig('./Figure/3.5/Feature_importance(split)_1.png', dpi = 600,bbox_inches = 'tight')

cols_5_1 = ['C', 'Si', 'Mn', 'Ni', 'Cr', 'Mo', 'V', 'Cu', 'W', 'Al', 'Ti', 'Nb', 'N', 'Co', 'B', 'P', 'S']
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 50,
         }
for i in (cols_5_1):
    print(i)
    pdp_goals = pdp.pdp_isolate(model=reg_ele, dataset=df_2, model_features=cols_5_1, feature=i, num_grid_points=10)
    plt.figure(figsize=(15, 10))
# 设置画布的尺寸
    pdp.pdp_plot(pdp_goals, i, figsize=(15, 10), center=True, ncols=30,
             plot_pts_dist=False, plot_lines=False, plot_params={
        # plot title and subtitle
        'title': 'PDP for feature "%s"' % i,
        'subtitle': None,
        'title_fontsize': 50,
        'subtitle_fontsize': None,
        'font_family': 'Arial',
        # matplotlib color map for ICE lines
        'line_cmap': 'Blues',
        'xticks_rotation': 0,
        # pdp line color, highlight color and line width
        'pdp_color': '#1A4E5D',
        'pdp_hl_color': '#FEDC00',
        'pdp_linewidth': 1.5,
        # horizon zero line color and with
        'zero_color': '#E75438',
        'zero_linewidth': 3,
        # pdp std fill color and alpha
        'fill_color': '#66C2D7',
        'fill_alpha': 0.5,
        # marker size for pdp line
        'markersize': 3.5,
      })

# plt.title(i,font1)#标题，并设定字号大小
    plt.tick_params(labelsize=50, grid_linewidth=1.5)
    plt.xlabel('')
    plt.savefig('./Figure/3.5/pdp_1D_nf/%s.png' % (i), dpi=600, bbox_inches='tight')
    plt.show()