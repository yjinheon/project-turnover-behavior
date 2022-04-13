from typing_extensions import Self
import seaborn as sns
import matplotlib.pyplot as plt


class Dataplots():
    """
    Data related plots
    
    - plot_corr_matrix
    - plot_outliers
    """
    
    def __init__():
        self.target = None
    
    def plot_outliers(self,colname):
        """
        series에서 이상치를 확인하고 이상치를 제거한 series를 반환
        """
        s=self.df[colname]
        
        try:
            f, ax = plt.subplots(2,2)
            
            sns.boxplot(s, orient='v', ax=ax[0,0]).set_title("With Outliers")
            sns.histplot(s, ax=ax[0,1]).set_title('With Outliers')
            sns.boxplot(self._remove_outliers(s), orient='v', ax=ax[1,0]).set_title("Without Outliers")
            sns.histplot(self._remove_outliers(s), ax=ax[1,1]).set_title('Without Outliers')
            f.tight_layout(w_pad=1.5,h_pad=1.0)
            

            plt.savefig(f'{s.name}.png')
        
        except ValueError as v:
            if s.dtype == 'object':
                raise Exception("This is not a numeric series")
            else:
                raise Exception(v)
            
    
    def plot_outliers_all(self):
        """
        df 전체의 numeric column에서 이상치를 확인하고 이상치 비교 plot 반환 
        """
        numeric_cols=self.numeric_df.columns.to_list()

        for col in numeric_cols:
            f, ax = plt.subplots(2,2)
            # subtitle
            f.suptitle("Outlier Analysis for {}".format(col))
            # orient 는 상자 세우기/눞히기의 차이
            sns.boxplot(self.df[col], orient='v', ax=ax[0][0]).set_title("With Outlier")
            sns.histplot(self.df[col], ax=ax[0][1]).set_title("With Outlier")
            sns.boxplot(self._remove_outliers(self.df[col]), orient='v', ax=ax[1][0]).set_title("Without Outlier")
            sns.histplot(self._remove_outliers(self.df[col]), ax=ax[1][1]).set_title("Without Outlier")
            f.tight_layout(w_pad=1.5,h_pad=1.0)
            
    plt.savefig(f'{colname}.png')
    plt.show()
    
    

"""
X, y 에 대한 plot
"""

class MLplots:
    
    def __init__(self):
        pass
    
    def plot_corr_matrix(self,X,y):
        """
        X, y 에 대한 correlation matrix plot
        """
        f, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
        plt.show()
    
    def plot_y(self,y):
        
        f, ax = plt.subplots(figsize=(10,10))
        sns.barplot(y.value_counts().index, y.value_counts().values, ax=ax)
        plt.show()
    
    
    def binary_class_plot(self,X,y,labels = ['0','1']):
        """
        binary class에 대한 confusion matrix와 roc curve
        """



def evalBinaryClassifier(model,x, y, labels=['Positives','Negatives']):
        '''
        Visualize the performance of  a Logistic Regression Binary Classifier.

        Displays a labelled Confusion Matrix, distributions of the predicted
        probabilities for both classes, the ROC curve, and F1 score of a fitted
        Binary Logistic Classifier. Author: gregcondit.com/articles/logr-charts

        Parameters
        ----------
        model : fitted scikit-learn model with predict_proba & predict methods
            and classes_ attribute. Typically LogisticRegression or 
            LogisticRegressionCV

        x : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples
            in the data to be tested, and n_features is the number of features

        y : array-like, shape (n_samples,)
            Target vector relative to x.

        labels: list, optional
            list of text labels for the two classes, with the positive label first

        Displays
        ----------
        3 Subplots

        Returns
        ----------
        F1: float
        '''
        #model predicts probabilities of positive class
        p = model.predict_proba(x)
        if len(model.classes_)!=2:
            raise ValueError('A binary class problem is required')
        if model.classes_[1] == 1:
            pos_p = p[:,1]
        elif model.classes_[0] == 1:
            pos_p = p[:,0]

        #FIGURE
        plt.figure(figsize=[15,4])

        #1 -- Confusion matrix
        cm = confusion_matrix(y,model.predict(x))
        plt.subplot(131)
        ax = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, 
                    annot_kws={"size": 14}, fmt='g')
        cmlabels = ['True Negatives', 'False Positives',
                'False Negatives', 'True Positives']
        for i,t in enumerate(ax.texts):
            t.set_text(t.get_text() + "\n" + cmlabels[i])
        plt.title('Confusion Matrix', size=15)
        plt.xlabel('Predicted Values', size=13)
        plt.ylabel('True Values', size=13)

        #2 -- Distributions of Predicted Probabilities of both classes
        df = pd.DataFrame({'probPos':pos_p, 'target': y})
        plt.subplot(132)
        plt.hist(df[df.target==1].probPos, density=True, 
                alpha=.5, color='green',  label=labels[0])
        plt.hist(df[df.target==0].probPos, density=True, 
                alpha=.5, color='red', label=labels[1])
        plt.axvline(.5, color='blue', linestyle='--', label='Boundary')
        plt.xlim([0,1])
        plt.title('Distributions of Predictions', size=15)
        plt.xlabel('Positive Probability (predicted)', size=13)
        plt.ylabel('Samples (normalized scale)', size=13)
        plt.legend(loc="upper right")

        #3 -- ROC curve with annotated decision point
        fp_rates, tp_rates, _ = roc_curve(y,p[:,1])
        roc_auc = auc(fp_rates, tp_rates)
        plt.subplot(133)
        plt.plot(fp_rates, tp_rates, color='green',
                lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
        #plot current decision point:
        tn, fp, fn, tp = [i for i in cm.ravel()]
        plt.plot(fp/(fp+tn), tp/(tp+fn), 'bo', markersize=8, label='Decision Point')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', size=13)
        plt.ylabel('True Positive Rate', size=13)
        plt.title('ROC Curve', size=15)
        plt.legend(loc="lower right")
        plt.subplots_adjust(wspace=.3)
        plt.show()
        #Print and Return the F1 score
        tn, fp, fn, tp = [i for i in cm.ravel()]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2*(precision * recall) / (precision + recall)
        printout = (
            f'Precision: {round(precision,2)} | '
            f'Recall: {round(recall,2)} | '
            f'F1 Score: {round(F1,2)} | '
        )
        print(printout)
        return F1