import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statistics import mode, mean, stdev
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from extra_tree import ExtraTreeForrest
from extra_diff_ev import ExtraDiffObliqueForrest
from time import time
import sklearn.preprocessing as preprocessing

from joblib import Parallel, delayed
import multiprocessing
 



class ExtraObliqueForrest:
    
    """
    alg_type = classification or regression, classification supported rn
    M = number of trees
    n_min = minimum sample size for splitting a node
    K  = the number of attributes randomly selected at each node
    """        
    def __init__(self, alg_type, M, n_min, K):
        self.used_combos = []
        self.alg_type = alg_type
        self.M = M
        self.n_min = n_min
        self.K = K
        self.entropy_beta = np.random.randint(2, 11)
    
    def fit(self, S, y=None):
        self.build_an_extra_tree_ensemble(S)
        open("eof_trees.txt",'w').write(str(self.trees))
    
    """
    S = dictionary of train data of the form: {header_value: data_list}
    returns a list of trees that can be used for classification / regression
    """        
    def build_an_extra_tree_ensemble(self, S):
        T = []
        S = self.to_dict(S)
        num_cores = multiprocessing.cpu_count()
             
        T = Parallel(n_jobs=num_cores)(delayed(self.build_an_extra_tree)(S) for i in range(self.M))
        # for i in range(self.M):
            # t = self.build_an_extra_tree(S)
            # T.append(t)
        self.trees = T
        return T
        
    """
    returns a tree of the form
    [{"['attr0', 'attr1'];[beta00, beta01];cm0": 
        [{"['attr2', 'attr3'];[beta10, beta11];cm1": 
                [[(class0, freq0)], 
                [(class1, freq1)]]}, 
        {"['attr4', 'atr5'];[beta20, beta21];cm2": 
                [[(classx, freq2)], 
                {"['attr6', 'attr7'];[beta30, beta31];cm3": 
                        [[(class0, freq3)], 
                        [(class1, freq4)]]
                }]
        }]
}]
    """
    def build_an_extra_tree(self, S):
        # print(len(S))
        if len(S) < self.n_min or self.all_attributes_are_constant(S):
            return self.labeled_leaf(S)
        score_beta = np.random.randint(low=2, high=11)
        a, beta, cm, sm = self.split_a_node(S, score_beta)
        if a is None:
            return self.labeled_leaf(S)
        S_l, S_r = self.get_splits(S, a, beta, cm, sm)
        if not S_l:
            t_r = self.build_an_extra_tree(S_r)
            return t_r
        if not S_r:
            t_l = self.build_an_extra_tree(S_l)
            return t_l
        t_l = self.build_an_extra_tree(S_l)
        t_r = self.build_an_extra_tree(S_r)
        t = {}
        t[str(a) +";"+ str(list(beta)) +";"+ str(cm)] = [t_l, t_r]
        return t
        
    """
    makes K random splits by K random attributes
    returns the attribute and the split_value that gave the best score
    """
    def split_a_node(self, S, beta):
        if self.stop_split(S):
            return None
        keys = list(S.keys())
        keys = self.check_keys(S)
        keys.remove('output')
        if len(keys) < self.K:
            return None, None, None, None
        a = [random.sample(keys, self.K) for _ in range(self.K * 3)]
        # print(len(a), len(self.used_combos))
        # a = [combo for combo in a if not self.is_used(combo)]
        # print(len(a), len(self.used_combos))
        s = [self.pick_a_random_split(S, ai) for ai in a]
        scores = [self.score(S, betai, ai, cm, sm, beta) for ai, betai, cm, sm in s]
        s_star = scores.index(min(scores))
        self.used_combos.append(s[s_star][0])
        # print(len(s[s_star]))
        return s[s_star]
        
    def is_used(self, combo):
        for used in self.used_combos:
            if len(set(used) & set(combo)) == len(combo):
                return True
        return False

    """
    returns a random value between the maximum and the minimum values of the attributes
    """
    def pick_a_random_split(self, S, a):
        mu, sigma = 0, 0.1 # mean and standard deviation
        beta = np.random.normal(mu, sigma, len(a))
        sm = []
        # mat = to_matrix(S)
        for k in range(len(S['output'])):
            cm = sum([S[key][k] * b for key, b in zip(a, beta)])
            sm.append(cm)
        cm_max = max(sm)
        cm_min = min(sm)
        if cm_max == cm_min:
            cm = cm_max
        else:
            cm = random.uniform(cm_min, cm_max)
        return a, beta, cm, sm
        
    def beta_entropy(self, target, beta):
        # beta = self.entropy_beta
        classes = set(target)
        lengths = [target.count(t) for t in classes]
        s = len(target)
        beta_ent = 1 / (1 - 2 ** (1 - beta))
        beta_ent *= (1 - sum([(bi / s) **  beta for bi in classes]))
        return beta_ent
        
        
    """
    purity gain
    """
    def score(self, S, si, ai, cm, sm, beta):
        #entropy
        hbs = self.beta_entropy(list(S['output']), beta)
        #split entropy
        left = [t for s, t in zip(sm, S['output']) if s < cm]
        right = [t for s, t in zip(sm, S['output']) if s > cm]
        hbb1 = self.beta_entropy(left, beta)
        hbb2 = self.beta_entropy(right, beta)
        split_etp = float(len(left))/len(S[ai[0]]) * hbb1 + float(len(right)) / len(S[ai[0]]) * hbb2
        #information gain
        ig = hbs - split_etp
        return 2 * ig / (hbs + split_etp)
          
    """
    gini impurity
    """
    def score2(self, S, si, ai, cm, sm, beta=0):
        left = [t for s, t in zip(sm, S['output']) if s < cm]
        right = [t for s, t in zip(sm, S['output']) if s > cm]
        etpl = self.entropy(left)
        etpr = self.entropy(right)
        return etpl + etpr
        
        
    def score1(self, S, si, ai, cm, sm, beta=0):
        #different for regression
        #entropy
        etp = self.entropy(list(S['output']))
        #split entropy
        left = [t for s, t in zip(sm, S['output']) if s < cm]
        right = [t for s, t in zip(sm, S['output']) if s > cm]
        etpl = self.entropy(left)
        etpr = self.entropy(right)
        split_etp = float(len(left))/len(S[ai[0]]) * etpl + float(len(right)) / len(S[ai[0]]) * etpr
        #information gain
        ig = etp - split_etp
        return 2 * ig / (etp + split_etp)
        return 2 * ig / (etp + split_etp)
        return ig


    """
    calculates the entropy of a list of binary values
    """    
    def entropy(self, target):
        # print('t', len(target))
        classes = set(target)
        # print('c', len(classes))
        ps = [target.count(t)/float(len(target)) for t in classes]
        # ps = [-pi * np.log2(pi) for pi in ps]
        ps = [pi * pi for pi in ps]
        return sum(ps)
        
    def get_entropies(self, df):
        ents = []
        for column in df:
            col = list(df[column])
            # print(type(col))
            # print(col)
            ents.append(self.entropy(col))
        return ents
           
    """
    splits a node in two based on the split value of the chosen attribute
    sm = the array of cms
    """
    def get_splits(self, S, a, beta, cm, sm):
        S_l = {}
        S_l = {}
        S_r = {}
        for i in range(len(S[a[0]])):
            for key in S:
                if sm[i] < cm:
                    if key not in S_l:
                        S_l[key] = []
                    S_l[key].append(S[key][i])
                else:
                    if key not in S_r:
                        S_r[key] = []
                    S_r[key].append(S[key][i])
        return S_l, S_r

    def check_keys(self, S):
        splittable_keys = []
        for key in S:
            verdict = False
            for i in range(len(S[key]) - 1):
                if S[key][i] != S[key][i + 1]:
                    verdict = True
            if verdict:
                splittable_keys.append(key)
        return splittable_keys
        
    """
    the node can no longer be split if all_attributes_are_constant or the node has too few attributes
    """
    def stop_split(self, S):
        # print(len(S))
        if len(S) < self.n_min:
            return True
        if self.all_attributes_are_constant(S):
            return True
        return False
        
    """
    returns True if all attributes are constant or the output variable is constant
    else returns False
    """
    def all_attributes_are_constant(self, S):
        verdict = True
        if not S:
            return verdict
        for i in range(len(S['output']) - 1):
            # print(S['output'])
            if S['output'][i] != S['output'][i + 1]:
                verdict = False
        if verdict == True:
            return True
        for attribute in S:
            for i in range(len(S[attribute]) - 1):
                if S[attribute][i] != S[attribute][i + 1]:
                    return False
        return True    
 
    def class_frequencies(self, S):
        from itertools import groupby
        if not S:
            return None
        a = S['output']
        freq = [(key, len(list(group)) / float(len(a))) for key, group in groupby(a)]
        freq = {x:a.count(x) / len(a) for x in a}
        freq = [(x, c) for x, c in freq.items()]
        return freq
        
    """
    returns a leaf labeled by class frequencies in S if alg type = classification
    returns a leaf labeled by average output in S if alg type = regression
    """
    def labeled_leaf(self, S):
        leaf = {}
        if self.alg_type == 'classification':
            leaf = self.class_frequencies(S)
            return leaf
        if self.alg_type == 'classification':
            leaf = avg(S['output'])
            return leaf  
        
    def predict(self, S):
        S = self.to_dict(S)
        mat, header = self.to_matrix(S)
        verdicts = []
        for line in mat:
            inst = {}
            for key, value in zip(header, line):
                inst[key] = value
            verdict = self.classify_instance(inst, self.trees)
            verdicts.append(verdict)
        return verdicts
        
    def classify_instance(self, S, trees):
        outputs = []
        for tree in trees:
            output = self.get_verdict(tree, S)
            if output is not None:
                outputs.append(output)
        # print(outputs)
        # return mode(outputs)
        return round(mean(outputs))
        
    def get_verdict(self, tree, S):
        if isinstance(tree, (list,)) and isinstance(tree[0], (tuple,)):
            max = 0
            output = None
            for tp in tree:
                if tp[1] > max:
                    max = tp[1]
                    output = tp[0]
            return output
            
        root = list(tree.keys())[0]
        import re
        (root_attr, root_beta, root_cm) = root.split(";")
        root_attr = re.split(", |'", root_attr.strip("[]"))
        root_beta =  re.split(", |'", root_beta.strip("[]"))
        root_attr = [elem for elem in root_attr if len(elem) > 0]
        root_beta = [float(elem) for elem in root_beta]
        # print(root_attr)
        root_cm = float(root_cm)
        root_value = sum([S[attr] * beta for attr, beta in zip(root_attr, root_beta)])
        if root_value < root_cm:
            return self.get_verdict(tree[root][0], S)
        else:
            return self.get_verdict(tree[root][1], S)

    def to_matrix(self, S):
        mat = []
        header = []
        for key in S:
            header.append(key)
            for i in range(len(S[key])):
                if len(mat) < i + 1:
                    mat.append([])
                mat[i].append(S[key][i])
        return mat, header  
            
    def to_dict(self, df):
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.mean())
        df = df.to_dict('split')
        S = {}
        for i, key in enumerate(df['columns']):
            S[key] = [row[i] for row in df['data']]
        return S
       
       
        
def get_poverty():
    data = pd.read_csv('.\\poverty\\train.csv')
    data = data.replace({'no': 0})
    data = data.replace({'yes': 1})
    # data = data.replace({'yes': 1})
    numeric_vars = [ 'v2a1', 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1',
                     'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv',
                     'escolari', 'rez_esc', 'hhsize', 'hogar_nin', 'hogar_adul',
                     'hogar_mayor', 'hogar_total', 'dependency', 'qmobilephone',
                     'meaneduc', 'bedrooms', 'overcrowding', 'age', 'Target']
    data = data[numeric_vars]
    df = data.rename(index=str, columns={"Target": "output"})
    
    df = df.replace({'output': {1: 0, 2: 0, 3: 0}})
    df = df.replace({'output': {4: 1}})
    # print(df.dtypes)
    df['dependency'] = df['dependency'].astype(float)
    return df
    
    
def get_diabetes():
    indian_data = pd.read_csv('.\\diabetes\\pima-indians-diabetes.csv')
    df = indian_data.rename(index=str, columns={"class": "output"})
    return df
    
    
def get_heart():
    heart_data = pd.read_csv('.\\heart_disease\\heart.csv')
    df = heart_data.rename(index=str, columns={"target": "output"})
    return df
    
    
def get_cancer():
    heart_data = pd.read_csv('.\\cancer\\data.csv')
    df = heart_data.rename(index=str, columns={'diagnosis': "output"})
    df = df.replace({'output': {'M': 1, 'B': 0}})
    df = df.drop('id', axis=1)
    df = df.fillna(0)   
    return df
    
    
def get_gratification():
    heart_data = pd.read_csv('.\\gratification\\train.csv')
    df = heart_data.rename(index=str, columns={"target": "output"})
    df = df.drop('id', axis=1)
    return df
    
    
def get_diabetes():
    indian_data = pd.read_csv('.\\diabetes\\pima-indians-diabetes.csv')
    df = indian_data.rename(index=str, columns={"class": "output"})
    return df
    
    
def get_titanic():
    titanic_data = pd.read_csv('.\\titanic\\train.csv')
    df = titanic_data.rename(index=str, columns={"Survived": "output"})
    df = df.drop('Name', axis=1)
    df = df.replace({'Sex': {'male': 0, 'female': 1}})
    df = df.replace({'Embarked': {'C': 0}})
    df = df.replace({'Embarked': {'S': 1}})
    df = df.replace({'Embarked': {'Q': 2}})
    df.Ticket = df.Ticket.str.extract('(\d+)')
    df['Ticket'] = df['Ticket'].astype(float)
    names = list(set(list(df.Cabin)))
    replace_set = list(set([name[0] for name in names[1:]]))
    replace_set = {elem: i + 1 for i, elem in enumerate(sorted(replace_set))}
    replace_dict = {}
    for i, name in enumerate(names[1:]):
        replace_dict[name] = replace_set[name[0]]
    
    df = df.replace({'Cabin': replace_dict})
    df = df.fillna(0)
    return df
    
    
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders
    
    
def get_income():
    income_data = pd.read_csv('.\\income\\adult.csv')
    income_data, _ = number_encode_features(income_data)
    df = income_data.rename(index=str, columns={"income": "output"})
    return df
    
    
def resample_df(df, type):
    df_one = df[df.output==1]
    df_two = df[df.output==0]
    no_train_samples = min([len(df_one), len(df_two)])
    df_one = resample(df_one, replace=True, n_samples=no_train_samples, random_state=123)
    df_two = resample(df_two, replace=True, n_samples=no_train_samples, random_state=123)
     
    df = pd.concat([df_one, df_two])
    df = df.sample(frac=1)
    print(df.output.value_counts())

    df = df.fillna(df.mean())
    y = df.output
    x = df.drop("output", axis=1)
    x = x.fillna(x.mean())
    y = y.fillna(y.mean())
    if type == 'ens':
        return df, y
    return x, y
    
    
def get_fam():         
    S = {'age': [20, 50, 49, 24, 22],
         'height': [172, 165, 175, 165, 183],
         'weight': [48, 80, 78, 57, 79],
         'gender': [1, 1, 2, 1, 2],
         'output': [1, 0, 0, 0, 1]}    
     
    inst =  {'age': 50,
         'height': 165,
         'weight': 80,
         'gender': 1} 
    return S, inst
    
def cv(name, ens, data, type='ens', splits=10):
    kf = KFold(n_splits=splits)
    t = time()
    y_true, y_pred = [], []
    for train_index, test_index in kf.split(data):
        df_train, df_test = data.iloc[train_index].fillna(0), data.iloc[test_index].fillna(0)
        y_train, y_test = df_train['output'].fillna(0), df_test['output'].fillna(0)
        y_true.extend(y_test)
        
        X_train, y_train = resample_df(df_train, type)
        trees = ens.fit(X_train, y_train)
        if type == 'sk':
            df_test = df_test.drop('output', axis=1)
        y_pred.extend(ens.predict(df_test))
        # print("bye")
        break
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(name, acc, time() - t)
    print(name, f1, time() - t)
    return acc, f1


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    ens = ExtraObliqueForrest('classification', 2, 2, 2)
    # seed = 5
    # random.seed(seed)
    init_df, samples = get_income(), 10000
    # x, y, df = resample_df(init_df, samples)
    # ents = ens.get_entropies(df)
    # ents.sort(reverse=True)
    # print("income", ents)
    init_df , samples= get_poverty(), 3500
    # x, y, df = resample_df(init_df, samples)
    # ents = ens.get_entropies(df)
    # ents.sort(reverse=True)
    # print("poverty", ents)
    # init_df, samples = get_diabetes(), 260
    # x, y, df = resample_df(init_df, samples)
    # ents = ens.get_entropies(df)
    # ents.sort(reverse=True)
    # print("diabetes", ents)
    init_df, samples = get_titanic(), 340
    # x, y, df = resample_df(init_df, samples)
    # ents = ens.get_entropies(df)
    # ents.sort(reverse=True)
    # print("titanic", ents)
    # init_df, samples = get_heart(), 130
    # x, y, df = resample_df(init_df, samples)
    # ents = ens.get_entropies(df)
    # ents.sort(reverse=True)
    # print("heart", ents)
    # init_df, samples = get_cancer(), 210
    # x, y, df = resample_df(init_df, samples)
    # ents = ens.get_entropies(df)
    # ents.sort(reverse=True)
    # print("cancer", ents)
    # x, y, y_true, df, df_test = resample_df(df)
    acc_eof = []
    acc_edof = []
    acc_etf = []
    acc_sketf = []
    acc_skerf = []
    
    # exit()
        
    
    # estimators = 11
    # min_samples_per_leaf = 20
    # max_no_features = 10
    
    estimators = 10
    min_samples_per_leaf = 5
    max_no_features = 2
    
    runs = 20
    
    for _ in range(runs):
        # norm
        df = init_df
        data=((df-df.min())/(df.max()-df.min()))
        data["output"]=df["output"]
        # x, y, df = resample_df(data, samples)
        
        
        ens = ExtraObliqueForrest('classification', estimators, min_samples_per_leaf, max_no_features)
        acc_eof.append(cv('eof', ens, data))
        
        t = time()
        
        # ens = ExtraDiffObliqueForrest('classification', estimators, min_samples_per_leaf, max_no_features)
        acc_edof.append(cv('edof', ens, data))
        
        
        t = time()
        ens = ExtraTreeForrest('classification', estimators, min_samples_per_leaf, max_no_features*3)
        acc_etf.append(cv('etf', ens, data))    
        
        ens = ExtraTreesClassifier(
        criterion="entropy",
        n_estimators=estimators, min_samples_leaf=min_samples_per_leaf, 
                                      max_features=max_no_features * 3)
        acc_sketf.append(cv('sketf', ens, data, type='sk'))
        
        ens = RandomForestClassifier(
        criterion="entropy",
        n_estimators=estimators, min_samples_leaf=min_samples_per_leaf, 
                                      max_features=max_no_features * 3)
                                     
        acc_skerf.append(cv('skerf', ens, data, type='sk'))
    
    f = open("accs.txt", "a")
    print("\nACC_FINAL:")
    print('edof', mean(acc_edof[:][0]), stdev(acc_edof[:][0]))
    print('eof', mean(acc_eof[:][0]), stdev(acc_eof[:][0]))
    print('etf', mean(acc_etf[:][0]), stdev(acc_etf[:][0]))
    print('sketf', mean(acc_sketf[:][0]), stdev(acc_sketf[:][0]))
    print('skerf', mean(acc_skerf[:][0]), stdev(acc_skerf[:][0]))
    
    print("\nF1_FINAL:")
    print('edof', mean(acc_edof[:][1]), stdev(acc_edof[:][1]))
    print('eof', mean(acc_eof[:][1]), stdev(acc_eof[:][1]))
    print('etf', mean(acc_etf[:][1]), stdev(acc_etf[:][1]))
    print('sketf', mean(acc_sketf[:][1]), stdev(acc_sketf[:][1]))
    print('skerf', mean(acc_skerf[:][1]), stdev(acc_skerf[:][1]))
            