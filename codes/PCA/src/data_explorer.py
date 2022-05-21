import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from data_util import ANALYSIS_COLUMNS
from dotenv import load_dotenv
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from bioinfokit.visuz import cluster

load_dotenv()
class DataExplorer(object):
    def __init__(self):
        wine_dir = os.getenv('DATA_DIR')
        self.train_df = pd.read_excel(wine_dir)

    def draw_pca(self):
        raw_df = self.train_df
        pca_data = self.get_pca()
        pca_df = pd.DataFrame(pca_data)
        pca_df.columns = [["pca1", "pca2"]]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Principal Component 1", fontsize=15)
        ax.set_ylabel("Principal Component 2", fontsize=15)
        ax.set_title("2 component PCA", fontsize=20)

        targets = ["A","B","C"]
        colors = ["r", "b","g"]
        for target, color in zip(targets, colors):
            indicesToKeep = raw_df["Type"] == target
            ax.scatter(
                pca_df.loc[indicesToKeep, "pca1"],
                pca_df.loc[indicesToKeep, "pca2"],
                c=color,
                s=50,
            )

        ax.legend(targets)
        ax.grid()
        plt.show()

    def draw_scree_plot(self):
        pca_data = self.get_pca(df_type='none')

        PC_values = np.arange(pca_data.n_components_) + 1
        plt.plot(PC_values, pca_data.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.show()

        return None

    def draw_loading_plot(self):
        pca = self.get_pca(df_type="df")
        g = sns.scatterplot(data=pca, x="pca1", y="pca2")
        self.loading_plot(coeff=pca[['pca1','pca2']].values,labels=pca.index,arrow_size=0.003)

        plt.show()

    def loading_plot(
            self, coeff, labels, scale=2, colors=None, visible=None, ax=plt, arrow_size=0.5
    ):
        for i, label in enumerate(labels):
            if visible is None or visible[i]:
                ax.arrow(
                    0,
                    0,
                    coeff[i, 0] * scale,
                    coeff[i, 1] * scale,
                    head_width=arrow_size * scale,
                    head_length=arrow_size * scale,
                    color="#001" if colors is None else colors[i],
                )
                ax.text(
                    coeff[i, 0] * 1.15 * scale,
                    coeff[i, 1] * 1.15 * scale,
                    label,
                    color="#001" if colors is None else colors[i],
                    ha="center",
                    va="center",
                )


    def get_pca(self, df_type="none"):
        pca = PCA(n_components=2)  # 주성분을 몇개로 할지 결정
        norm_X = self.__get_X(self.train_df)

        if df_type=="none":
            printcipalComponents = pca.fit(np.nan_to_num(norm_X))
            return printcipalComponents
        else:
            printcipalComponents = pca.fit_transform(np.nan_to_num(norm_X))
            principalDf = pd.DataFrame(data=printcipalComponents, columns=["pca1", "pca2"])
            return principalDf

    def __get_X(self, X):
        X = X[ANALYSIS_COLUMNS].values
        X_mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        norm_X = np.divide(np.subtract(X, X_mean), std)
        return norm_X

    def show_column_name(self):
        return self.train_df.columns


if __name__ == "__main__":

    de = DataExplorer()
    print(de.draw_loading_plot())
