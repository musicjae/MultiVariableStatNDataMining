from dataGenerator import DataGenarator
from models import SVMer
from sklearn.metrics import precision_score,recall_score
def main(mode):
    dg = DataGenarator()
    models = SVMer()

    ds = dg.build_dataset()
    features = ds[['v1','v2','v3','v4','v5']].values
    labels = ds['label'].values

    if mode == "lof":
        predictions = models.get_lof(features)
        precision = precision_score(predictions,labels)
        recall = recall_score(predictions,labels)
        return precision, recall
    elif mode == "ocs":
        predictions = models.get_Ocsvm(features)
        precision = precision_score(predictions,labels)
        recall = recall_score(predictions,labels)
        return precision, recall
    elif mode == "ifore":
        predictions = models.get_iForest(features)
        precision = precision_score(predictions,labels)
        recall = recall_score(predictions,labels)
        return precision, recall

if __name__ == '__main__':
    print(main(mode="ifore"))