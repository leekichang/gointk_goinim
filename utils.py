import numpy as np
import sklearn.metrics as metrics


def printLearningData(epoch, EPOCH, AVG_LOSS_TRAIN, AVG_LOSS_VAL, ACC_VAL):
    print(f'(epoch {epoch+1 : 03}/{EPOCH: 03}) | Training Loss : {AVG_LOSS_TRAIN:.5f} | ',
          f'Validation Loss :{AVG_LOSS_VAL:.5f} | Validation Accuracy : {ACC_VAL*100:.2f} %', sep = '')

def get_metrics(pred, anno, n_label, plot=False):
    print(np.shape(pred))
    print(np.shape(anno))
    print(metrics.accuracy_score(anno, pred))
    conf_mat = metrics.confusion_matrix(anno, pred)
    print(conf_mat)
    print(metrics.classification_report(anno, pred))

    if plot == True:
        plt.rcParams["figure.figsize"] = (n_label, n_label)
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.show()
        conf_mat_sum = np.sum(conf_mat, axis=1)
        conf_mat_sum = np.reshape(conf_mat_sum, (n_label, 1))
        sns.heatmap(conf_mat/conf_mat_sum, annot=True, fmt='.2%', cmap='Blues')