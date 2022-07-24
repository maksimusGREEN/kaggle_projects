from tqdm import tqdm
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML
import torch

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

class FeatureSelector:

    def __init__(
        self,
        estimator:BaseEstimator,
        train_data:pd.DataFrame,
        val_data:pd.DataFrame,
        *,
        target_col:str,
        cols_to_drop:list=None,
        cat_features:list=None
    ):
        """
        data: pd.DataFrame
            Полное признаковое пространство (без целевой переменной)
        target: list like
            Целевая переменная
        """
        
        self.train = train_data
        self.val = val_data
        self.cols_to_drop = cols_to_drop
        self._estimator = clone(estimator)
        self.target_col = target_col
        self.cat_features = list(set(cat_features).intersection(set(train_data.columns)))
        self.logs=dict()
        
        if not np.allclose(self.train.index.values, np.arange(0, self.train.shape[0])):
            raise AssertionError('Индекс датафрейма должен начинаться с 0 и заканчиваться dataframe.shape[0] - 1 (значения по порядку)')

        if not isinstance(self._estimator, (CatBoostClassifier, LGBMClassifier, CatBoostRegressor, LGBMRegressor)):
            raise AssertionError(f'Class FeatureSelector supports only CatBoost or LGBM, not {type(self._estimator)}')

    def drop_useless_cols(self):
        # Первый этап отбора признаков
        train_X, train_y = self.train.drop((self.cols_to_drop+[self.target_col]), axis=1),  self.train[self.target_col]
        val_X, val_y = self.val.drop((self.cols_to_drop+[self.target_col]), axis=1),  self.val[self.target_col]

        
        
        if isinstance(self._estimator, (LGBMClassifier, LGBMRegressor)):
            
            model_selector = clone(self._estimator)

            if self.cat_features:
                _ = model_selector.fit(train_X,
                                   train_y,
                                   eval_set=(val_X, val_y),
                                   eval_names=('train', 'valid'),
                                   categorical_feature=self.cat_features,
                                   verbose=-1
                                )
            else:
                _ = model_selector.fit(train_X,
                                   train_y,
                                   eval_set=(val_X, val_y),
                                   eval_names=('train', 'valid'),
                                   verbose=-1
                                )
            output = pd.DataFrame(zip(model_selector.booster_.feature_name() ,model_selector.feature_importances_), columns=['features', 'importance'])
        elif isinstance(self._estimator, (CatBoostClassifier, CatBoostRegressor)):
            
            model_selector = self._estimator.copy()
            
            if self.cat_features:
                cat_ind = [i for i, col in enumerate(train_X.columns) if col in self.cat_features]
                train_pool = Pool(train_X, train_y, cat_features=cat_ind)
                val_pool = Pool(val_X, val_y, cat_features=cat_ind)
                model_selector.fit(train_pool, eval_set=val_pool, silent=True)
            else:
                train_pool = Pool(train_X, train_y)
                val_pool = Pool(val_X, val_y)
                model_selector.fit(train_pool, eval_set=val_pool, silent=True)
            output = pd.DataFrame(zip(model_selector.feature_names_ ,model_selector.feature_importances_), columns=['features', 'importance'])
        else:
            raise AssertionError(f'Class FeatureSelector supports only CatBoostClassifier or LGBMClassifier, not {type(self._estimator)}')


        cols = output[np.invert(output.importance>0)].features.values.tolist()
        self.train =  self.train.drop(cols, axis=1)
        self.val =  self.val.drop(cols, axis=1)
        self.cat_features = [col for col in self.cat_features if col in self.train.columns]
        feature_imp = output.set_index('features').importance
        self.__feat_imp = list(feature_imp[feature_imp>0].sort_values(ascending=False).index)
        print('Удалено признаков с нулевой значимостью: {}'.format(len(cols)))
        print('Осталось признаков: {}'.format(len(self.__feat_imp)))
        return self


    def get_best_features(self, bigger_is_better:bool, custom_metric_lgbm=None, n_iter=None, verbose=True):
        
        def compare_two(a, b, bigger_is_better):
            if bigger_is_better:
                return a>b
            else:
                return a<b
        
        best_feat = self.__feat_imp[:5]
        candidats = self.__feat_imp[5:]
        target_name = self.target_col
        cat_features = list(self.cat_features)    
            
        if not n_iter:
            iter_ = len(candidats)//2-1
        else:
            iter_= max(n_iter, len(candidats)//2-1)


        if isinstance(self._estimator, (CatBoostClassifier, CatBoostRegressor)):
            
            model_selector = self._estimator.copy()
            
            if model_selector.get_param('eval_metric'):
                metric_name = model_selector.get_param('eval_metric')
            else:
                if model_selector.get_param('loss_function'):
                    metric_name=model_selector.get_param('loss_function')
                else:
                    clf = self._estimator.copy()
                    _ = clf.fit(self.train[best_feat], self.train[target_name], silent=True)
                    metric_name=clf.get_all_params()['eval_metric']
            
            if cat_features:

                cat_ind = [best_feat.index(cat_features[i]) for i in range(len(cat_features)) if cat_features[i] in best_feat] 

                train_pool = Pool(self.train[best_feat], self.train[target_name], cat_features=cat_ind)
                val_pool = Pool(self.val[best_feat], self.val[target_name], cat_features=cat_ind)
            else:
                train_pool = Pool(self.train[best_feat], self.train[target_name])
                val_pool = Pool(self.val[best_feat], self.val[target_name])                   

            hist_ = model_selector.fit(
                train_pool,
                eval_set=val_pool,
                silent=True) 
            best_score = model_selector.best_score_['validation'][metric_name]

            BS  = best_score
            cur_feature_set = best_feat.copy()    
            self.logs['baseline'] = f'Baseline: {len(best_feat)} features and {str(best_score)[:6]} score'

            d = 0
            
            for iteration in tqdm(range(iter_), position=0, leave=True):
                cur_feature_set  += candidats[iteration*2:iteration*2+2]
                model_selector = self._estimator.copy()

                if cat_features:
                    cat_ind = [cur_feature_set.index(cat_features[i]) for i in range(len(cat_features)) if cat_features[i] in cur_feature_set]
                    train_pool = Pool(self.train[cur_feature_set], self.train[target_name], cat_features=cat_ind)
                    val_pool = Pool(self.val[cur_feature_set], self.val[target_name], cat_features=cat_ind)

                else:
                    train_pool = Pool(self.train[cur_feature_set], self.train[target_name])
                    val_pool = Pool(self.val[cur_feature_set], self.val[target_name])                        

                hist_ = model_selector.fit(
                    train_pool,
                    eval_set=val_pool,
                    silent=True) 

                new_score = model_selector.best_score_['validation'][metric_name]

                if compare_two(new_score, best_score, bigger_is_better):
                    d += 1
                    best_score = new_score
                    best_feat = cur_feature_set.copy()
                    self.logs.setdefault('iterations', [])\
                        .append(f'iteration: {iteration+1} of {iter_}| added: {candidats[iteration*2:iteration*2+2]} | new set of features: {len(best_feat)} of {len(best_feat+candidats)} | new score: {str(best_score)[:6]}')
                    if d==4:
                        d = 0
                        cols = best_feat[:-2*4]
                        to_remove = 0
                        for j,f in enumerate(cols):
                            cur_feature_set.remove(f)
                            model_selector = self._estimator.copy()

                            if cat_features:
                                cat_ind = [cur_feature_set.index(cat_features[i]) for i in range(len(cat_features)) if cat_features[i] in cur_feature_set]
                                train_pool = Pool(self.train[cur_feature_set], self.train[target_name], cat_features=cat_ind)
                                val_pool = Pool(self.val[cur_feature_set], self.val[target_name], cat_features=cat_ind)                                    

                            else:
                                train_pool = Pool(self.train[cur_feature_set], self.train[target_name])
                                val_pool = Pool(self.val[cur_feature_set], self.val[target_name])                                        

                            hist_ = model_selector.fit(
                                train_pool,
                                eval_set=val_pool,
                                silent=True)

                            new_score = model_selector.best_score_['validation'][metric_name]

                            if compare_two(new_score,best_score, bigger_is_better):
                                best_score = new_score
                                best_feat = cur_feature_set.copy()
                                self.logs.setdefault('iterations', [])\
                                    .append(f'iteration: {iteration+1} of {iter_}| removed: {f} | new set of features: {len(best_feat)} of {len(best_feat+candidats)} | new score: {str(best_score)[:6]}')
                                break
                            else:
                                    cur_feature_set.append(f)       
                else:
                    cur_feature_set = cur_feature_set[:-2]
           
        elif isinstance(self._estimator, (LGBMClassifier, LGBMRegressor)):
            
            model_selector = clone(self._estimator)
            
            if not custom_metric_lgbm:
            
                if cat_features:
            
                    cat_f = [col for col in cat_features if col in best_feat]


                    _ = model_selector.fit(self.train[best_feat],
                                       self.train[target_name],
                                       verbose=False,
                                       categorical_feature=cat_f,
                                       eval_set=[(self.train[best_feat], self.train[target_name]), (self.val[best_feat],self.val[target_name])],
                                       eval_names=['train', 'valid']
                                      )

                    best_score = list(model_selector.best_score_['valid'].values())[-1]
                else:
                    _ = model_selector.fit(self.train[best_feat],
                                       self.train[target_name],
                                       verbose=False,
                                       eval_set=[(self.train[best_feat], self.train[target_name]), (self.val[best_feat],self.val[target_name])],
                                       eval_names=['train', 'valid']
                                      )

                    best_score = list(model_selector.best_score_['valid'].values())[-1]

                BS  = best_score
                cur_feature_set = best_feat.copy()
                self.logs['baseline'] = f'Baseline: {len(best_feat)} features and {str(best_score)[:6]} score'    

                d = 0
                for iteration in tqdm(range(iter_), position=0, leave=True):
                    cur_feature_set  += candidats[iteration*2:iteration*2+2]
                    model_selector = clone(self._estimator)
                    if cat_features:
                        cat_f = [col for col in cat_features if col in cur_feature_set]
                        _ = model_selector.fit(self.train[cur_feature_set],
                                           self.train[target_name],
                                           verbose=False,
                                           categorical_feature=cat_f,
                                           eval_set=[(self.train[cur_feature_set], self.train[target_name]), (self.val[cur_feature_set],self.val[target_name])],
                                           eval_names=['train', 'valid']
                                  )
                    else:
                        _ = model_selector.fit(self.train[cur_feature_set],
                                               self.train[target_name],
                                               verbose=False,
                                               eval_set=[(self.train[cur_feature_set], self.train[target_name]), (self.val[cur_feature_set],self.val[target_name])],
                                               eval_names=['train', 'valid']
                                              )

                    new_score = list(model_selector.best_score_['valid'].values())[-1]
                    if compare_two(new_score,best_score, bigger_is_better):
                        d += 1
                        best_score = new_score
                        best_feat = cur_feature_set.copy()
                        self.logs.setdefault('iterations', [])\
                            .append(f'iteration: {iteration+1} of {iter_}| added: {candidats[iteration*2:iteration*2+2]} | new set of features: {len(best_feat)} of {len(best_feat+candidats)} | new score: {str(best_score)[:6]}')
                        if d==4:
                            d = 0
                            cols = best_feat.copy()[:-2*4]
                            to_remove = 0

                            for j,f in enumerate(cols):
                                cur_feature_set.remove(f)
                                if cat_features:
                                    cat_f = [col for col in cat_features if col in cur_feature_set]
                                    _ = model_selector.fit(self.train[cur_feature_set],
                                                           self.train[target_name],
                                                           verbose=False,
                                                           categorical_feature=cat_f,
                                                           eval_set=[(self.train[cur_feature_set], self.train[target_name]), (self.val[cur_feature_set],self.val[target_name])],
                                                           eval_names=['train', 'valid']
                                                      )
                                    new_score = list(model_selector.best_score_['valid'].values())[-1]

                                if compare_two(new_score,best_score, bigger_is_better):
                                    best_score = new_score
                                    best_feat = cur_feature_set.copy()
                                    self.logs.setdefault('iterations', [])\
                                        .append(f'iteration: {iteration+1} of {iter_}| removed: {f} | new set of features: {len(best_feat)} of {len(best_feat+candidats)} | new score: {str(best_score)[:6]}')
                                    break
                                else:
                                    cur_feature_set.append(f)       
                    else:
                        cur_feature_set = cur_feature_set[:-2]
            else:
                if cat_features:
                    
                    cat_f = [col for col in cat_features if col in best_feat]
                    model_selector.fit(self.train[best_feat],
                                       self.train[target_name],
                                       verbose=False,
                                       categorical_feature=cat_f,
                                       eval_set=[(self.train[best_feat], self.train[target_name]), (self.val[best_feat],self.val[target_name])],
                                       eval_names=['train', 'valid'],
                                       eval_metric=custom_metric_lgbm
                                      )

                    best_score = list(model_selector.best_score_['valid'].values())[-1]
                else:
                    _ = model_selector.fit(self.train[best_feat],
                                       self.train[target_name],
                                       verbose=False,
                                       eval_set=[(self.train[best_feat], self.train[target_name]), (self.val[best_feat],self.val[target_name])],
                                       eval_names=['train', 'valid'],
                                       eval_metric=custom_metric_lgbm
                                      )

                    best_score = list(model_selector.best_score_['valid'].values())[-1]

                BS  = best_score
                cur_feature_set = best_feat.copy()    

                d = 0
                for iteration in tqdm(range(iter_), position=0, leave=True):
                    cur_feature_set  += candidats[iteration*2:iteration*2+2]
                    if cat_features:
                        cat_f = [col for col in cat_features if col in cur_feature_set]
                        _ = model_selector.fit(self.train[cur_feature_set],
                                               self.train[target_name],
                                               verbose=False,
                                               categorical_feature=cat_f,
                                               eval_set=[(self.train[cur_feature_set], self.train[target_name]), (self.val[cur_feature_set],self.val[target_name])],
                                               eval_names=['train', 'valid'],
                                               eval_metric=custom_metric_lgbm
                                      )
                    else:
                        _ = model_selector.fit(self.train[cur_feature_set],
                                               self.train[target_name],
                                               verbose=False,
                                               eval_set=[(self.train[cur_feature_set], self.train[target_name]), (self.val[cur_feature_set],self.val[target_name])],
                                               eval_names=['train', 'valid'],
                                               eval_metric=custom_metric_lgbm
                                              )

                    new_score = list(model_selector.best_score_['valid'].values())[-1]
                    if compare_two(new_score,best_score, bigger_is_better):
                        d += 1
                        best_score = new_score
                        best_feat = cur_feature_set.copy()
                        self.logs.setdefault('iterations', [])\
                            .append(f'iteration: {iteration+1} of {iter_}| added: {candidats[iteration*2:iteration*2+2]} | new set of features: {len(best_feat)} of {len(best_feat+candidats)} | new score: {str(best_score)[:6]}')
                        if d==4:
                            d = 0
                            cols = best_feat.copy()[:-2*4]
                            to_remove = 0

                            for j,f in enumerate(cols):
                                cur_feature_set.remove(f)
                                if cat_features:
                                    cat_f = [col for col in cat_features if col in cur_feature_set]
                                    _ = model_selector.fit(self.train[cur_feature_set],
                                                           self.train[target_name],
                                                           verbose=False,
                                                           categorical_feature=cat_f,
                                                           eval_set=[(self.train[cur_feature_set], self.train[target_name]), (self.val[cur_feature_set],self.val[target_name])],
                                                           eval_names=['train', 'valid'],
                                                           eval_metric=custom_metric_lgbm
                                                      )
                                    new_score = list(model_selector.best_score_['valid'].values())[-1]

                                if compare_two(new_score,best_score,bigger_is_better):
                                    best_score = new_score
                                    best_feat = cur_feature_set.copy()
                                    self.logs.setdefault('iterations', [])\
                                        .append(f'iteration: {iteration+1} of {iter_}| removed: {f} | new set of features: {len(best_feat)} of {len(best_feat+candidats)} | new score: {str(best_score)[:6]}')
                                    break
                                else:
                                    cur_feature_set.append(f)       
                    else:
                        cur_feature_set = cur_feature_set[:-2]
        else:
            raise AssertionError(f'Class FeatureSelector supports only CatBoostClassifier or LGBMClassifier, not {type(self._estimator)}')

        if verbose:
            print(self.logs['baseline'])
            print('--'*20)
            for log in self.logs['iterations']:
                print(log, end="\n")
            print('--'*20)
            if bigger_is_better:
                print(f'final set of features: {len(best_feat)}, -{str((1-len(best_feat)/len(best_feat+candidats))*100)[:4]}% reduction of initial set | new score: {str(best_score)[:6]}, +{str((best_score/BS-1)*100)[:4]}% improvement vs initial')
            else:
                print(f'final set of features: {len(best_feat)}, -{str((1-len(best_feat)/len(best_feat+candidats))*100)[:4]}% reduction of initial set | new score: {str(best_score)[:6]}, {str((best_score/BS-1)*100)[:4]}% improvement vs initial')
        #add final removal
        return best_feat


def make_automl_prediction(train:pd.DataFrame,
                           test:pd.DataFrame,
                           cols_to_drop:list=None,
                           *,
                           task_type:str=None,
                           loss:str = None,
                           target_name:str=None,
                           **kwargs):

    np.random.seed(kwargs.get('RANDOM_STATE', 42))
    torch.set_num_threads(kwargs.get('N_THREADS', 4))

    task = Task(name = task_type, loss = loss)

    roles = {
            'target': target_name,
            'drop': cols_to_drop
            }

    automl = TabularAutoML(
                            task = task, 
                            timeout = kwargs.get('TIMEOUT', 600),
                            cpu_limit = kwargs.get('N_THREADS', 6),
                            reader_params = {'n_jobs': kwargs.get('N_THREADS', 6), 'cv':kwargs.get('N_FOLDS', 4), 'random_state': kwargs.get('RANDOM_STATE', 42)}
                        )
    _ = automl.fit_predict(train, roles = roles)
    return (np.exp(automl.predict(test).data.flatten())//1000)*1000



def make_catboost_prediction(estimator:BaseEstimator,
                             train:pd.DataFrame,
                             valid:pd.DataFrame,
                             test:pd.DataFrame=None,
                             cat_features:list=None,
                             return_model:bool=False,
                             cols_to_drop:list=[],
                             *,
                             target_name:str=None,
                             ):
    if not isinstance(estimator, (CatBoostRegressor, CatBoostClassifier)):
        raise AssertionError(f'estimator should be CatBoostRegressor or CatBoostClassifier, not {type(estimator)}')
    
    assert train.shape[1]==valid.shape[1]

    if not estimator.is_fitted():

        if cat_features:
            cat_f = [col for col in cat_features if col in train.columns]

            train_pool = Pool(train.drop(cols_to_drop+[target_name], axis=1), label=train[target_name], cat_features=cat_f)
            valid_pool = Pool(valid.drop(cols_to_drop+[target_name], axis=1), label=valid[target_name], cat_features=cat_f)
        else:
            train_pool = Pool(train.drop(cols_to_drop+[target_name], axis=1), label=train[target_name])
            valid_pool = Pool(valid.drop(cols_to_drop+[target_name], axis=1), label=valid[target_name])
        estimator.fit(X=train_pool, eval_set=valid_pool, verbose_eval=250, use_best_model=True)

    if return_model:
        if test is not None:
            return (np.exp(estimator.predict(test.drop(cols_to_drop+[target_name], axis=1)))//1000)*1000, estimator
        else:
            return (np.exp(estimator.predict(valid.drop(cols_to_drop+[target_name], axis=1)))//1000)*1000, estimator
    else:
        if test is not None:
            return (np.exp(estimator.predict(test.drop(cols_to_drop+[target_name], axis=1)))//1000)*1000
        else:
            return (np.exp(estimator.predict(valid.drop(cols_to_drop+[target_name], axis=1)))//1000)*1000

def preproc_for_classic_ml(data:pd.DataFrame, cat_cols:list=None, cols_to_drop:list=None):
    if cols_to_drop:
        data_classic = data.drop(cols_to_drop, axis=1)
    else:
        data_classic = data.copy()
    if not cat_cols:
        cat_cols = data.select_dtypes(include=object).columns
    cat_f = [col for col in cat_cols if col in data_classic.columns]
    for column in cat_f:
        data_classic[column] = data_classic[column].astype('category').cat.codes
    cols_for_dummy = []
    for column in cat_f:
        if 2<data_classic[column].nunique()<=16:
            cols_for_dummy.append(column)
    if cols_for_dummy:
        data_classic = pd.get_dummies(data_classic, columns=cols_for_dummy, dummy_na=False)
    
    if 'ownership' in data_classic.columns:
        data_classic['ownership'] = data_classic.ownership.fillna(data_classic.query('sample=="train"').ownership.median())
    
    if 'enginedisplacement' in data_classic.columns:
        data_classic['enginedisplacement'] = data_classic.enginedisplacement.fillna(0)
        
    return data_classic



def make_lgbm_prediction(estimator:BaseEstimator,
                         train:pd.DataFrame,
                         valid:pd.DataFrame,
                         test:pd.DataFrame=None,
                         cat_features:list=None,
                         return_model:bool=False,
                         cols_to_drop:list=[],
                         *,
                         target_name:str=None,
                         is_fitted:bool=False,
                         **kwargs
                             ):
    if not isinstance(estimator, (LGBMClassifier, LGBMRegressor)):
        raise AssertionError(f'estimator should be LGBMClassifier or LGBMRegressor, not {type(estimator)}')
    
    assert train.shape[1]==valid.shape[1]

    if not is_fitted:
        
        train_X, train_y = train.drop(cols_to_drop+[target_name], axis=1), train[target_name]
        valid_X, valid_y = valid.drop(cols_to_drop+[target_name], axis=1), valid[target_name]

        if cat_features is not None:
            cat_f = [col for col in cat_features if col in train_X.columns]

            estimator.fit(X=train_X,
                        y=train_y,
                        eval_set=[(train_X, train_y),(valid_X, valid_y)],
                        eval_names=('train', 'valid'),
                        categorical_feature=cat_f,
                        **kwargs)
        else:
            estimator.fit(X=train_X,
                        y=train_y,
                        eval_set=[(train_X, train_y),(valid_X, valid_y)],
                        eval_names=('train', 'valid'),
                        **kwargs)

    if return_model:    
    
        if test is not None:
            return (np.exp(estimator.predict(test.drop(cols_to_drop+[target_name], axis=1))))//1000*1000, estimator
        else:
            return (np.exp(estimator.predict(valid_X)))//1000*1000, estimator
    
    else:

        if test is not None:
            return (np.exp(estimator.predict(test.drop(cols_to_drop+[target_name], axis=1))))//1000*1000
        else:
            return (np.exp(estimator.predict(valid_X)))//1000*1000

def preproc_for_mlp(data:pd.DataFrame,
                    cat_cols:list=None,
                    num_cols:list=None,
                    cols_to_drop:list=None,
                    scaler=StandardScaler,
                    *,
                    target:str=None,
                    random_state=None)->pd.DataFrame:
    if cols_to_drop:
        data_mlp = data.drop(cols_to_drop, axis=1)
    else:
        data_mlp = data.copy()
    if not num_cols:
        num_cols = data_mlp.select_dtypes(include='number').drop(target,axis=1).columns
        num_cols = [column for column in num_cols if data_mlp[column].nunique()>2]
    if not cat_cols:
        cat_cols = data_mlp.select_dtypes(include=object).columns
    cat_f = [col for col in cat_cols if col in data_mlp.columns]
    for column in cat_f:
        data_mlp[column] = data_mlp[column].astype('category').cat.codes
    cols_for_dummy = [column for column in cat_f if data_mlp[column].nunique()>2]
    if cols_for_dummy:
        data_mlp = pd.get_dummies(data_mlp, columns=cols_for_dummy, dummy_na=False)

    if 'enginedisplacement' in data_mlp.columns:
        data_mlp['enginedisplacement'] = data_mlp.enginedisplacement.fillna(0)

    train, valid = train_test_split(data_mlp.query("sample=='train'").drop('sample', axis=1), test_size=0.2, shuffle=True, random_state=random_state)
    test = data_mlp.query("sample=='test'").drop('sample', axis=1)
    
    if 'ownership' in data_mlp.columns:
        fill_value = train.ownership.median()
        train['ownership'] = train.ownership.fillna(fill_value)
        valid['ownership'] = valid.ownership.fillna(fill_value)
        test['ownership'] = test.ownership.fillna(fill_value)

    #Scaling
    scaler = scaler()
    train.loc[:, num_cols] = scaler.fit_transform(train[num_cols])
    valid.loc[:, num_cols] = scaler.transform(valid[num_cols])
    test.loc[:, num_cols] = scaler.transform(test[num_cols])
    
    return train, valid, test

