import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def analysis_glass():
    from sklearn.cross_validation import train_test_split
    fp = "glass.csv"
    df = pd.read_csv(fp)
    train, test = train_test_split(df, test_size = 0.3)
    train = train.as_matrix()
    train_X = train[:, 0:9]
    train_Y = train[:,9]
    
    test = test.as_matrix()
    test_X = test[:, 0:9]
    test_Y = test[:,9]
    

    return train_X, train_Y, test_X, test_Y

def do_gradient_boost(lr = 1.0, md = 1):
    #The best values of lr and md have to be determined through grid search
    # for this dataset ~ lr =0.05, md =3 gave 0.769 on the test set
    from sklearn.ensemble import GradientBoostingClassifier

    train_X, train_Y, test_X, test_Y = analysis_glass()

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,\
                                     max_depth=md, \
                                     random_state=0).fit(train_X, train_Y)
    
    return clf.score(test_X, test_Y)

def do_random_forests(md = None):
    from sklearn.ensemble import RandomForestClassifier
    train_X, train_Y, test_X, test_Y = analysis_glass()
    rfc = RandomForestClassifier(n_estimators=100, max_depth = md)
    rfc.fit(train_X, train_Y)

    return rfc.score(test_X, test_Y)

# Try extremely randomized trees
def do_extra_trees(md = None):
    from sklearn.ensemble import ExtraTreesClassifier
    train_X, train_Y, test_X, test_Y = analysis_glass()
    ETC = ExtraTreesClassifier(n_estimators=100, max_depth = md)
    ETC.fit(train_X, train_Y)

    return ETC.score(test_X, test_Y)

# Lets try a combination of Random Tree Embedding and Naive Bayes

def  do_TRT(ne = 10, md = 3):
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.naive_bayes import BernoulliNB
    train_X, train_Y, test_X, test_Y = analysis_glass()
    all_X = np.vstack((train_X, test_X))
    hasher = RandomTreesEmbedding(n_estimators=ne,\
                                  random_state=0, max_depth=md)
    all_X_trans = hasher.fit_transform(all_X)
    train_X_trans = all_X[0:149, :]
    test_X_trans = all_X[149:, :]

    nb = BernoulliNB()
    nb.fit(train_X_trans, train_Y)

    return nb.score(test_X_trans, test_Y)

    
    
    
    
    
