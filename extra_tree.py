import random
import numpy as np
import pandas as pd
from statistics import mode

from joblib import Parallel, delayed
import multiprocessing

class ExtraTreeForrest:

    def __init__(self, alg_type, M, n_min, K):
        self.used_combos = []
        self.alg_type = alg_type
        self.M = M
        self.n_min = n_min
        self.K = K
        self.entropy_beta = np.random.randint(2, 11)

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
            return None, None
        a = random.sample(keys, self.K)
        s = [self.pick_a_random_split(S, ai) for ai in a]
        scores = [self.score(S, si, ai, beta) for ai, si in s]
        s_star = scores.index(max(scores))
        return s[s_star]

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
    calculates the entropy of a list of values
    """    
    def entropy(self, target):
        classes = set(target)
        ps = [target.count(t)/float(len(target)) for t in classes]
        ps = [-pi * np.log2(pi) for pi in ps]
        return sum(ps)
        
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
    def score1(self, S, si, ai, beta):
        #entropy
        hbs = self.beta_entropy(list(S['output']), beta)
        #split entropy
        left = [t for a, t in zip(S[ai], S['output']) if a < si]
        right = [t for a, t in zip(S[ai], S['output']) if a > si]
        hbb1 = self.beta_entropy(left, beta)
        hbb2 = self.beta_entropy(right, beta)
        split_etp = float(len(left))/len(S[ai]) * hbb1 + float(len(right)) / len(S[ai]) * hbb2
        #information gain
        ig = hbs - split_etp
        return 2 * ig / (hbs + split_etp)
        
    """
    normalized version of the shannon information gain
    """
    def score(self, S, si, ai, beta=0):
        #different for regression
        #entropy
        etp = self.entropy(list(S['output']))
        #split entropy
        left = [t for a, t in zip(S[ai], S['output']) if a < si]
        right = [t for a, t in zip(S[ai], S['output']) if a > si]
        etpl = self.entropy(left)
        etpr = self.entropy(right)
        split_etp = float(len(left))/len(S[ai]) * etpl + float(len(right)) / len(S[ai]) * etpr
        #information gain
        ig = etp - split_etp
        # return 2 * ig / (etp + split_etp)
        return ig
              
    """
    gini impurity
    """
    def score2(self, S, si, ai, beta=0):
        left = [t for a, t in zip(S[ai], S['output']) if a < si]
        right = [t for a, t in zip(S[ai], S['output']) if a > si]
        etpl = self.entropy(left)
        etpr = self.entropy(right)
        return etpl + etpr
        
    """
    returns a random value between the maximum and the minimum values of the attributes
    """
    def pick_a_random_split(self, S, a):
        a_max = max(S[a])
        a_min = min(S[a])
        if a_max == a_min:
            a_c = a_max
        else:
            a_c = random.uniform(a_min, a_max)
        return a, a_c
        
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
    
    def fit(self, S, y=None):
        self.build_an_extra_tree_ensemble(S)  
            
    """
    S = dictionary of train data of the form: {header_value: data_list}
    alg_type = classification or regression, classification supported rn
    M = number of trees
    n_min = minimum sample size for splitting a node
    K  = the number of attributes randomly selected at each node
    returns a list of trees that can be used for classification / regression
    """        
    def build_an_extra_tree_ensemble(self, S):
        T = []
        S = self.to_dict(S)
        num_cores = multiprocessing.cpu_count()
             
        T = Parallel(n_jobs=num_cores)(delayed(self.build_an_extra_tree)(S) for i in range(self.M))
        self.trees = T
        return T

    """
    returns a tree of the form
    {
        (attr0, split_val1): 
            [
                [(output0, freq0)], 
                {
                    (attr1, split_val1): 
                        [
                            [(output1, freq1)], 
                            [(output2, freq2)]
                        ]
                }
            ]
    }
    """
    def build_an_extra_tree(self, S):
        # print(len(S))
        if len(S) < self.n_min or self.all_attributes_are_constant(S):
            return self.labeled_leaf(S)
        beta = np.random.randint(2, 11)
        a, s_star = self.split_a_node(S, beta)
        if a is not None:
            S_l, S_r = self.get_splits(S, a, s_star)
            if not S_l:
                t_r = self.build_an_extra_tree(S_r)
                return t_r
            if not S_r:
                t_l = self.build_an_extra_tree(S_l)
                return t_l
            t_l = self.build_an_extra_tree(S_l)
            t_r = self.build_an_extra_tree(S_r)
            t = {}
            t[(a, s_star)] = [t_l, t_r]
            return t
        else:
            return self.labeled_leaf(S)
        
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
        
    """
    splits a node in two based on the split value of the chosen attribute
    """
    def get_splits(self, S, a, s_star):
        S_l = {}
        S_r = {}
        for i in range(len(S[a])):
            for key in S:
                if key != a:
                    if S[a][i] < s_star:
                        if key not in S_l:
                            S_l[key] = []
                        S_l[key].append(S[key][i])
                    else:
                        if key not in S_r:
                            S_r[key] = []
                        S_r[key].append(S[key][i])
        return S_l, S_r
        
    def classify_instance(self, S, trees):
        outputs = []
        for tree in trees:
            output = self.get_verdict(tree, S)
            if output is not None:
                outputs.append(output)
        # print(outputs)
        return mode(outputs)
        
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
        (root_attr, root_cut) = root
        if S[root_attr] < root_cut:
            return self.get_verdict(tree[root][0], S)
        else:
            return self.get_verdict(tree[root][1], S)
            
    def to_dict(self, df):
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.mean())
        df = df.to_dict('split')
        S = {}
        for i, key in enumerate(df['columns']):
            S[key] = [row[i] for row in df['data']]
        return S
   
if __name__ == "__main__":   
    import pandas as pd

    data = pd.read_csv('.\\train.csv')
    data = data.replace({'no': 0})
    # data = data.replace({'yes': 1})
    numeric_vars = [ 'v2a1', 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1',
                     'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv',
                     'escolari', 'rez_esc', 'hhsize', 'hogar_nin', 'hogar_adul',
                     'hogar_mayor', 'hogar_total', 'dependency', 'qmobilephone',
                     'meaneduc', 'bedrooms', 'overcrowding', 'age', 'Target']
    data = data[numeric_vars]
    df = data.rename(index=str, columns={"Target": "output"})


    #resample
    from sklearn.utils import resample
    df = df.replace({'output': {1: 0, 2: 0, 3: 0}})
    df = df.replace({'output': {4: 1}})
    # indian_data = pd.read_csv('.\\pima-indians-diabetes.csv')
    # df = indian_data.rename(index=str, columns={"class": "output"})
    print(df['output'].value_counts())
    df_one = df[df.output==1]
    df_two = df[df.output==0]

    no_samples = 2000
    df_one_test = resample(df_one, replace=True, n_samples=no_samples, random_state=23)
    df_two_test = resample(df_two, replace=True, n_samples=no_samples, random_state=23)
    df_one = resample(df_one, replace=True, n_samples=no_samples, random_state=123)
    df_two = resample(df_two, replace=True, n_samples=no_samples, random_state=123)
     
    df = pd.concat([df_one, df_two])
    df_test = pd.concat([df_one_test, df_two_test])
    df = df.sample(frac=1)
    print(df.output.value_counts())

    import sys
    sys.setrecursionlimit(3000)

    data = to_dict(df)

    # trees = build_an_extra_tree_ensemble(data, 'classification', 11, 5, 2)
    trees = build_an_extra_tree_ensemble(data, 'classification', 21, 20, 10)

    from sklearn.metrics import accuracy_score
    test_data = to_dict(df_test)
    y_pred = classify(test_data, trees)
    # print(y_pred)
    y_true = test_data['output']
    # print(y_true)
    print(accuracy_score(y_true, y_pred))
           
        
    S = {'age': [20, 50, 49, 24, 22],
         'height': [172, 165, 175, 165, 183],
         'weight': [48, 80, 78, 57, 79],
         'gender': [1, 1, 2, 1, 2],
         'output': [1, 0, 2, 0, 1]}    
     
    inst =  {'age': 20,
         'height': 172,
         'weight': 48,
         'gender': 1} 

    # trees = build_an_extra_tree_ensemble(S, 'classification', 1, 2, 2)
    # print(classify_instance(inst, trees))
        
        