import random
import numpy as np
import pandas as pd
from statistics import mode, mean
from sklearn.utils import resample
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from extra_tree import ExtraTreeForrest
from time import time
import sklearn.preprocessing as preprocessing

from joblib import Parallel, delayed
import multiprocessing
 



class ExtraDiffObliqueForrest:
    
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
        self.f = random.uniform(0, 2)
        self.entropy_beta = np.random.randint(2, 11)
    
    def fit(self, S, y=None):
        self.build_an_extra_tree_ensemble(S)
        open("edof_trees.txt",'w').write(str(self.trees))
    
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
    def pick_a_random_split(self, S, a, trials=5):
        mu, sigma = 0, 0.1 # mean and standard deviation
        cr = 0.002
        cr = 0.002
        population = [(np.random.normal(mu, sigma, len(a)), 0) for _ in range(trials)]
        
        for _ in range(trials):
            new_population = []
            for beta, old_cms in population:
                sm = []
                new_sm = []
                xa = np.random.normal(mu, sigma, len(a))
                xb = np.random.normal(mu, sigma, len(a))
                xc = np.random.normal(mu, sigma, len(a))
                mutated_beta = self.f * (xa - xb) + xc
                new_beta = []
                for elem_old, elem_new in zip(beta, new_beta):
                    r = random.random()
                    if r <= cr:
                        new_beta.append(elem_new)
                    else:
                        new_beta.append(elem_old)
                    
                sm = []
                for k in range(len(S['output'])):
                    cm = sum([S[key][k] * b for key, b in zip(a, beta)])
                    sm.append(cm)
                cm_max = max(sm)
                cm_min = min(sm)
                if cm_max == cm_min:
                    cm = cm_max
                else:
                    cm = random.uniform(cm_min, cm_max)
                    
                new_sm = []
                for k in range(len(S['output'])):
                    cm = sum([S[key][k] * b for key, b in zip(a, new_beta)])
                    new_sm.append(cm)
                cm_max = max(new_sm)
                cm_min = min(new_sm)
                if cm_max == cm_min:
                    new_cm = cm_max
                else:
                    new_cm = random.uniform(cm_min, cm_max)
                if self.score(S, beta, a, cm, sm) > self.score(S, new_beta, a, new_cm, new_sm):    
                    new_population.append((new_beta, new_cm))
                else:
                    new_population.append((beta, cm))
            population = new_population
        result = []
        for beta, old_cm in population:
            sm = []
            for k in range(len(S['output'])):
                cm = sum([S[key][k] * b for key, b in zip(a, beta)])
                sm.append(cm)
            cm_max = max(sm)
            cm_min = min(sm)
            if old_cm == 0:
                if cm_max == cm_min:
                    cm = cm_max
                else:
                    cm = random.uniform(cm_min, cm_max)
            else:
                cm = old_cm
            result.append([a, beta, cm, sm])
        scores = [self.score(S, betai, ai, cm, sm) for ai, betai, cm, sm in result]
        local_s_star = scores.index(min(scores))
        return result[local_s_star]
    
    def beta_entropy(self, target, beta):
        classes = set(target)
        lengths = [target.count(t) for t in classes]
        s = len(target)
        beta_ent = 1 / (1 - 2 ** (1 - beta))
        beta_ent *= (1 - sum([(bi / s) **  beta for bi in classes]))
        return beta_ent
        
        
    """
    purity gain
    """
    def score1(self, S, si, ai, cm, sm, beta):
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
        
        
    def score(self, S, si, ai, cm, sm, beta=0):
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
        # return 2 * ig / (etp + split_etp)
        return ig


    """
    calculates the entropy of a list of binary values
    """    
    def entropy(self, target):
        classes = set(target)
        ps = [target.count(t)/float(len(target)) for t in classes]
        # ps = [-pi * np.log2(pi) for pi in ps]
        ps = [pi * pi for pi in ps]
        return sum(ps)
           
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
    print(df.head())
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
    print(df.head())
    return df
    
    
def resample_df(df, no_samples = 2000):
    print(df['output'].value_counts())
    df_one = df[df.output==1]
    df_two = df[df.output==0]

    no_test_samples = int(0.25 * no_samples)
    no_train_samples = int(0.75 * no_samples)
    df_one_test = resample(df_one, replace=True, n_samples=no_test_samples, random_state=23)
    df_two_test = resample(df_two, replace=True, n_samples=no_test_samples, random_state=23)
    df_one = resample(df_one, replace=True, n_samples=no_train_samples, random_state=123)
    df_two = resample(df_two, replace=True, n_samples=no_train_samples, random_state=123)
     
    df = pd.concat([df_one, df_two])
    df_test = pd.concat([df_one_test, df_two_test])
    df = df.sample(frac=1)
    print(df.output.value_counts())
                              
    y_true = df_test['output']
    df = df.fillna(df.mean())
    df_test = df_test.fillna(df_test.mean())
    df_test = df_test.drop("output", axis=1)
    y = df.output
    x = df.drop("output", axis=1)
    return x, y, y_true, df, df_test
    
    
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


if __name__ == "__main__":
    seed = 5
    random.seed(seed)
    df = get_income()
    # exit()
    # df = get_poverty()
    # df = get_diabetes()
    # df = get_titanic()
    # df = get_gratification()
    # df = get_heart()
    # x, y, y_true, df, df_test = resample_df(df)
    x, y, y_true, df, df_test = resample_df(df, 10000)
    
    
    # mine oblique
    t = time()
    # ens = ExtraObliqueForrest('classification', 21, 20, 10)
    ens = ExtraObliqueForrest('classification', 11, 5, 2)
    trees = ens.build_an_extra_tree_ensemble(df)

    test_data = ens.to_dict(df_test)
    y_pred = ens.classify(test_data, trees)
    print("Eot: ", accuracy_score(y_true, y_pred), time() - t)
    
    
    # mine extra
    t = time()
    # et = ExtraTreeForrest('classification', 21, 20, 10)
    et = ExtraTreeForrest('classification', 11, 5, 2)
    trees = et.build_an_extra_tree_ensemble(df)
    test_data = et.to_dict(df_test)
    y_pred = et.classify(test_data, trees)
    print("Et: ", accuracy_score(y_true, y_pred), time() - t)
    
    
    # sk extra
    t = time()
    # forest = ExtraTreesClassifier(n_estimators=21, min_samples_leaf=20, 
                                  # max_features=10, random_state=seed)
    forest = ExtraTreesClassifier(n_estimators=11, min_samples_leaf=5, 
                                  max_features=2, random_state=seed)
    forest.fit(x, y)
    y_pred_sk = forest.predict(df_test)
    
    print("Sklearn et: ", accuracy_score(y_true, y_pred_sk), time() - t)

    # ens = ExtraObliqueForrest('classification', 1, 2, 2)
    # trees = ens.build_an_extra_tree_ensemble(S)
    # y_pred = ens.classify_instance(inst, trees)
    # print(y_pred)
        