from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
class SVMer(object):

    def get_lof(self,x):
        clf = LocalOutlierFactor(n_neighbors=5)
        reseult = clf.fit_predict(x)

        return reseult

    def get_iForest(self,x):
        clf = IsolationForest(random_state=827,max_features=0.7).fit_predict(x)
        return clf

    def get_Ocsvm(self,x):
        clf = OneClassSVM(gamma='auto').fit_predict(x)

        return clf

    def get_Svdd(self):

        return None