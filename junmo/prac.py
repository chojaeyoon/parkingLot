"""
    Written on July 15, 2021 by Junmo Kim
    
    First Try

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from tqdm import tqdm



rawtrain = pd.read_csv('train.csv')
rawtest = pd.read_csv('test.csv')
age_gender = pd.read_csv('age_gender_info.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv')


"""
    지역정보 Join (운전가능연령: 20~70대)
    
"""
female = []
male = []
for col in age_gender.columns:
    female.append(col) if '여자' in col else male.append(col)
        
female_ratio = pd.DataFrame(age_gender[female].iloc[:, 2:8].sum(axis=1), columns=['여자운전비율'])
male_ratio = pd.DataFrame(age_gender[male].iloc[:, 2:8].sum(axis=1), columns=['남자운전비율'])

rawtrain = pd.merge(rawtrain, female_ratio, on='지역')
rawtrain = pd.merge(rawtrain, male_ratio, on='지역')
rawtest = pd.merge(rawtest, female_ratio, on='지역')
rawtest = pd.merge(rawtest, male_ratio, on='지역')


def exception():
    """
        오류 단지코드 제외

        1. 전용면적별 세대수 합계와 총세대수 일치 x: 전용면적별 세대수 합계를 총 세대수로 설정하고 진행
        2. 동일한 단지에 단지코드가 2개로 부여된 경우: train(2085, 1397, 2431, 1649, 1036), test(2675)

    """
    # 1. 전용면적별 세대수 합계 join
    _t1 = rawtrain.groupby('단지코드').sum()['전용면적별세대수']
    _t2 = pd.merge(rawtrain, _t1, on='단지코드')
    _t2['전용면적별세대수'] = _t2['전용면적별세대수_x']
    _t2['총세대수'] = _t2['전용면적별세대수_y']
    _final1 = _t2.drop(['전용면적별세대수_x', '전용면적별세대수_y'], axis='columns')
    
    # 2. 제외할 코드들 제외
    train = _final1[~_final1['단지코드'].isin(['C2085', 'C1397', 'C2431', 'C1649', 'C1036', 'C1095', 
                                             'C2051', 'C1218', 'C1894', 'C2483', 'C1502', 'C1988'])]
    test = rawtest[~rawtest['단지코드'].isin(['C2675', 'C2335', 'C1327'])]
    
    return train, test

train, test = exception()


def nanprocess_v1(usetrain):
    """
        결측치 처리
    
        도보 10분거리내 지하철역 수: "지역별로" 결측치 제외 평균 0.5 미만이면 0, 이상이면 1 부여
        cf. 대전과 충청남도에만 결측치 있음    
        
    """
    cities = list(set(usetrain['지역']))
    aparts = list(set(usetrain['단지코드']))
    _pre1 = usetrain.copy()
    
    _col = '도보 10분거리 내 지하철역 수(환승노선 수 반영)'
    for city in tqdm(cities):
        _aparts_in_city = usetrain[usetrain['지역'] == city]
        _codes_in_city = list(set(_aparts_in_city['단지코드']))
        
        _mean = np.nanmean(_aparts_in_city[_col])
        for code in _codes_in_city:
            _idx = _pre1[_pre1['단지코드']==code].index
            if usetrain[usetrain['단지코드'].isin(_codes_in_city)][_col].isnull().sum() > 0:
                if _mean >= 0.5:
                    _pre1.loc[_idx, _col] = 1
                else:
                    _pre1.loc[_idx, _col] = 0
                    
    """
        임대보증금 및 임대료: 결측치 제외하고, 단지별로, 전용면적과 선형회귀분석 진행하여 결측치 처리
        cf. 임대보증금, 임대료에 '-' 값 있음: 결측치로 처리 
        cf. C2152 아파트의 경우 임대보증금/임대료 데이터가 아예 결측 -> 강원도평균으로 처리
        
    """        
    # 임대가치 지표 신규 설정
    _pre2 = _pre1.copy()
    _col = ['임대보증금', '임대료']

    for col in _col:
        _pre2 = _pre2.drop(_pre2[_pre2[col] == '-'].index)
        _pre2[col] = _pre2[col].astype(float)
    _pre2['임대가치'] = _pre2['임대보증금'] * _pre2['임대료']
    
    
#     # 이상치 제거, 선형회귀식 작성에 사용할 데이터 추출
#     q1 = _pre1['임대가치'].quantile(0.25)
#     q3 = _pre1['임대가치'].quantile(0.75)
#     IQR = q3 - q1
#     _lin = _pre1[(_pre1['임대가치'] < (q1 - 1.5 * IQR)) | (_pre1['임대가치'] > (q3 + 1.5 * IQR))]
    
#     # 선형회귀식 작성
#     cities = list(set(_pre1['지역']))
#     aparts = list(set(_pre1['단지코드']))
    
#     for apart in tqdm(aparts):
#         _dat1 = _lin[_lin['단지코드'] == apart]
#         linmodel = stats.linregress(list(_lin['전용면적'].astype(float)), list(_lin['임대가치'].astype(float)))
#         _idx1 = _pre1[_pre1['단지코드'] == apart].isnull().index
#         for idx in _idx1:
#             _pre1.loc[idx, '임대가치'] = linmodel.intercept + linmodel.slope * _pre1.loc[idx, '전용면적']
    
#     final = _pre1[(_pre1['임대가치'] >= (q1 - 1.5 * IQR)) & (_pre1['임대가치'] <= (q3 + 1.5 * IQR))]
    
    return _pre1, _pre2

_, prep_train = nanprocess_v1(train)
_forC2152, prep_test = nanprocess_v1(test)

# 강원도 임대가치 평균으로 C2152 결측치 처리
_forC2152['임대가치'] = np.mean(prep_test[prep_test['지역']=='강원도']['임대가치'])
prep_test = pd.concat([prep_test, _forC2152[_forC2152['단지코드'] == 'C2152']])

# 테스트 결측데이터 처리
prep_test.loc[400, '자격유형'] = 'A'
prep_test.loc[599, '자격유형'] = 'C'

prep = pd.concat([prep_train, prep_test])



def preprocess_v1(prep, type='train'):
    """
        String 데이터 / 상가 데이터 처리

    """
    # 지역
    local_map = {}
    for i, loc in enumerate(prep['지역'].unique()):
        _arr = [0] * len(prep['지역'].unique())
        _arr[i] = 1
        local_map[loc] = _arr

    # 공급유형
    supply_map = {}
    for i, loc in enumerate(prep['공급유형'].unique()):
        _arr = [0] * len(prep['공급유형'].unique())
        _arr[i] = 1
        supply_map[loc] = _arr

    # 자격유형
    qual_map = {}
    for i, loc in enumerate(prep['자격유형'].unique()):
        _arr = [0] * len(prep['자격유형'].unique())
        _arr[i] = 1
        qual_map[loc] = _arr
        
    
    aparts = list(set(prep['단지코드']))
    merge_set = []
    for code in tqdm(aparts):
        final_vector = {}

        usedat = prep[prep['단지코드'] == code]
        onlyapart = usedat[usedat['임대건물구분'] == '아파트']
        
        if '상가' in set(usedat['임대건물구분']):
            sanga = 1
            sangadat = usedat[usedat['임대건물구분'] == '상가']
            apartdat = usedat[usedat['임대건물구분'] == '아파트']
            sanga_area = sum(sangadat['전용면적'] * sangadat['전용면적별세대수'])
            apart_area = sum(apartdat['전용면적'] * apartdat['전용면적별세대수'])
        else:
            sanga = 0
            sanga_area = 0.0
            apart_area = sum(usedat['전용면적'] * usedat['전용면적별세대수'])
        
        final_vector['단지코드'] = [usedat['단지코드'].iloc[0]]
        final_vector['총세대수'] = [usedat['총세대수'].iloc[0]]
        final_vector['상가'] = [sanga]
        final_vector['아파트면적'] = [apart_area]
        final_vector['상가면적'] = [sanga_area]
        
        _onehot = sum([np.array(local_map[key]) for key in usedat['지역'].unique()])    # 지역정보
        for tp in zip(list(local_map.keys()), list(_onehot)):
            final_vector[tp[0]] = tp[1]
            
        _onehot = sum([np.array(supply_map[key]) * usedat.iloc[idx]['전용면적별세대수'] for idx, key in enumerate(usedat['공급유형'])])    # 공급유형
        for tp in zip(supply_map.keys(), _onehot):
            final_vector[tp[0]] = tp[1]
            
        _onehot = sum([np.array(qual_map[key]) * usedat.iloc[idx]['전용면적별세대수'] for idx, key in enumerate(usedat['자격유형'])])    # 자격유형
        for tp in zip(qual_map.keys(), _onehot):
            final_vector[tp[0]] = tp[1]     

        final_vector['공가수'] = [usedat['공가수'].iloc[0]]            
        final_vector['임대가치'] = [usedat['임대가치'].iloc[0]]
        final_vector['지하철'] = [usedat['도보 10분거리 내 지하철역 수(환승노선 수 반영)'].iloc[0]]
        final_vector['버스'] = [usedat['도보 10분거리 내 버스정류장 수'].iloc[0]]
        final_vector['주차면수'] = [usedat['단지내주차면수'].iloc[0]]
        if type == 'train':
            final_vector['등록차량수'] = [usedat['등록차량수'].iloc[0]]
        
        del final_vector['공공분양']
        
        merge_set.append(pd.DataFrame(final_vector))
    
    return pd.concat(merge_set)

finaltrain = preprocess_v1(prep_train).dropna()
finaltest = preprocess_v1(prep_test, 'test')




"""
    Normalization

"""
means = {}
stds = {}
for col in finaltrain.columns:
    means[col] = np.mean(finaltrain[col])
    stds[col] = np.std(finaltrain[col])
    finaltrain[col] = (finaltrain[col] - means[col]) / stds[col]
    
for col in finaltest.columns:
    finaltest[col] = (finaltest[col] - means[col]) / stds[col]



"""
    학습 시작

"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error as MAE

import time


X = finaltrain[finaltrain.columns.difference(['등록차량수'])]
y = finaltrain[['등록차량수']]

def myLR():
    model = LinearRegression()
    fit = model.fit(X, y)
    print(f'Linear Regression Score: {MAE(y, fit.predict(X))}')


def myRegressor(regressor, param_grid):
    start = time.time()
    reg_grid = GridSearchCV(estimator=regressor,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=20,
                            cv=5,
                            refit=True,
                            return_train_score=True)
    reg_grid.fit(X, y)
    result = pd.DataFrame(reg_grid.cv_results_)[
        ['params', 'mean_test_score', 'rank_test_score']
    ].sort_values(by='rank_test_score')
    print(f'소요시간: {round((time.time() - start) / 60, 2)}분')
    
    return result


rf_params = {
    'n_estimators': [10, 20, 30, 40, 50],
    'criterion': ['mae'],
    'max_depth': [20, 30, 40]
}

svr_params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'gamma': ['scale', 'auto'],
    'C': [0.1, 1, 2, 3],
    'epsilon': [0.01, 0.1, 0.5],
}


# result_rf = myRF(rf_params)
result_svr = myRegressor(SVR(), svr_params)







