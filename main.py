import pandas as pd
import numpy as np

# загрузка данных
train_prepared = pd.read_csv('TRAIN_PREPARED.csv')
train_additional_data = pd.read_csv('TRAIN_ADDITIONAL_DATA.csv')

# группировка  последних действий пользователей по email (остальные данные не использую)
data = train_additional_data.groupby(['email'], as_index=False).last()

# данные столбцов 'action' и 'trial' перевожу в числовой вид, учитывая что в 'trial' есть nan
create = pd.get_dummies(data.action, drop_first=True)
trial = pd.get_dummies(data.trial, dummy_na=True, drop_first=True)
data = data.join(create.join(trial))
data.columns = ['email', 'action', 'trial', 'calday', 'action_dummies', 'trial_dummies', 'trial_nan']

# объединяю информацию из предыдущего блока с основным датасетом
train_prepared = train_prepared.merge(data[['email', 'action_dummies', 'trial_dummies', 'trial_nan']], how='left')

# формирую данные для обучения модели
X = train_prepared.iloc[:,1].to_frame().join(train_prepared.iloc[:,5:145].join(train_prepared.iloc[:,146:149]))
y = train_prepared.label

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, make_scorer

# разбиваем данные на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# использую random forest, т.к. градиентный бустинг даже на гугл колаб очень долго обучается.
# глубину деревьев подобрал экспериментально, балансировка весов клвссов исходя из соотношения классов в обучающей выборке
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=0, class_weight={0: 1, 1: 4})

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict_proba(X_test)

# подозрительно высокий скор
roc_auc_score(y_test, y_pred[:,1])    # 0.9906380611720469

# загрузка тестовых данных
test_prepared = pd.read_csv('TEST_PREPARED.csv')
test_additional_data = pd.read_csv('TEST_ADDITIONAL_DATA.csv')

# действия с датасетами по аналогии с train
test_data = test_additional_data.groupby(['email'], as_index=False).last()

test_create = pd.get_dummies(test_data.action, drop_first=True)
test_trial = pd.get_dummies(test_data.trial, dummy_na=True, drop_first=True)
test_data = test_data.join(test_create.join(test_trial))
test_data.columns = ['email', 'action', 'trial', 'calday', 'action_dummies', 'trial_dummies', 'trial_nan']

test_prepared = test_prepared.merge(test_data[['email', 'action_dummies', 'trial_dummies', 'trial_nan']], how='left')

X_submit = test_prepared.iloc[:,1].to_frame().join(test_prepared.iloc[:,5:149])

# прогнозирование вероятности принадлежности к классу отток на тестовых данных
y_submit = rf_clf.predict_proba(X_submit)

# запись результата
result_data = pd.DataFrame({'email': test_prepared.email, 'prediction': np.around(y_submit[:,1], 3)})
result_data.to_csv('submit.csv', index=False)
