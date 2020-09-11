from itertools import cycle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


def generate_train_test(ml_method, data_version, data_process, period):
    if ml_method == 'tsc':
        training_dataset = h5py.File('data_set/{}.h5'.format(data_version), 'r')
    elif ml_method == 'mlp':
        training_dataset = h5py.File('data_set/mlp_{}.h5'.format(data_version), 'r')
    else:
        raise ValueError('The data version is not recognized')
        
    x_train = training_dataset['x_{}'.format(period)][()]
    y_train = training_dataset['y_{}'.format(period)][()]
    x_test = training_dataset['x_2019'][()]
    y_test = training_dataset['y_2019'][()]
    
    if 'no_flux' in data_process:
        # remove the last 5 features
        x_train = x_train[..., :-5]
        x_test = x_test[..., :-5]
        
    if 'fusion' in data_process:
        y_train[y_train == 7] = 2
        y_train[y_train == 9] = 2
        #y_train[y_train == 8] = 0
        y_train[y_train == 8] = 7

        y_test[y_test == 7] = np.int8(2)
        y_test[y_test == 9] = np.int8(2)
        #y_test[y_test == 8] = np.int8(0)
        y_test[y_test == 8] = np.int8(7)
    
    if 'shuffle' in data_process:
        x_total =  np.concatenate((x_train, x_test), axis=0)
        y_total =  np.concatenate((y_train, y_test), axis=0)

        x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.25, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)
        
        if 'oversampling' in data_process:
            x_train_foreshock = x_train[y_train == 5]
            y_train_foreshock = y_train[y_train == 5]
            print(x_train_foreshock.shape)

            x_train = np.concatenate((x_train, x_train_foreshock))
            y_train = np.concatenate((y_train, y_train_foreshock))

            x_train, x_empty, y_train, y_empy = train_test_split(x_train, y_train, test_size=0.001, random_state=1)
            
    else:
        x_val = x_test
        y_val = y_test
        
    if 'minmax' in data_process:
        x_train_normalize = normalize_ts(x_train, x_train)
        x_val_normalize = normalize_ts(x_val, x_train)
        x_test_normalize = normalize_ts(x_test, x_train)
        
        x_train = x_train_normalize
        x_val = x_val_normalize
        x_test = x_test_normalize

    elif 'znorm' in data_process:
        x_train_normalize = standardize_ts(x_train, x_train)
        x_val_normalize = standardize_ts(x_val, x_train)
        x_test_normalize = standardize_ts(x_test, x_train)

        x_train = x_train_normalize
        x_val = x_val_normalize
        x_test = x_test_normalize

 
    return x_train, y_train, x_test, y_test, x_val, y_val


def generate_output_path(ml_method, classifier_name, experiment, itr):
    tsc_root_dir = 'dl-4-tsc/dl-tsc-temp'
    mlp_root_dir = 'mlp'
    if ml_method == 'tsc':
        output_directory = tsc_root_dir + '/results/' + classifier_name + '/sitl_ml' + itr + '/' + experiment + '/'
    elif ml_method == 'mlp':
        output_directory = mlp_root_dir + '/results/' + classifier_name + '/mlp' + itr + '/' + experiment + '/'
    else:
        raise ValueError('not yet implemented')
    
    done_path = output_directory + 'DONE'
    
    return output_directory, done_path


def normalize_ts(ts_input, ts_train):
    min_output = 0.01
    max_output= 0.99
    output = []
    
    nb_feature = ts_input.shape[-1]
    for feature_index in np.arange(nb_feature):
        max_value = ts_train[:, :, feature_index].max()
        min_value = ts_train[:, :, feature_index].min()
        zero_one_output = (ts_input[:, :, feature_index] - min_value) / (max_value - min_value)
        output.append(min_output + max_output * zero_one_output)
    
    normalized_ts = np.stack(output, axis=2)
    return normalized_ts
    

def standardize_ts(ts_input, ts_train):
    output = []

    nb_feature = ts_input.shape[-1]
    for feature_index in np.arange(nb_feature):
        mean_value = np.mean(ts_train[:, :, feature_index])
        std_value = np.std(ts_train[:, :, feature_index])
        standardize_output = (ts_input[:, :, feature_index] - mean_value) / (std_value)
        output.append(standardize_output)

    normalized_ts = np.stack(output, axis=2)
    return normalized_ts

    
def scale_input_data(x_train, x_val):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_ = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    return x_train_, x_val

        
def confusion_matrix(y_pred, y_true, association=False):
    data = {'y_pred': y_pred, 'y_true': y_true}
    df = pd.DataFrame(data, columns=['y_true','y_pred'])
    confusion_matrix = pd.crosstab(df['y_pred'], df['y_true'], rownames=['Predicted'], colnames=['True'])
    if association:
        confusion_matrix.rename(columns=association, inplace=True)
        confusion_matrix.rename(index=association, inplace=True)

        confusion_array = confusion_matrix.to_numpy()
        
        percentage = np.diag(confusion_array) / (np.sum(confusion_array, axis=0 ) + np.sum(confusion_array, axis=1))

    return confusion_matrix, percentage


def roc_plot(y_score_proba, y_true, association=False):
    # Plot linewidth.
    lw = 2.0

    classes_list = np.unique(y_true)
    n_classes = len(classes_list)
    
    print('n_classes', n_classes)
    
    y_true_roc = label_binarize(y_true, classes=classes_list)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_roc[:, i], y_score_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_roc.ravel(), y_score_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure(1)

    full_list = ['aqua', 'darkorange', 'cornflowerblue', 'limegreen', 'firebrick', 'orchid']
    if len(full_list) > n_classes:
        colors = cycle(full_list[:n_classes])
    else:
        colors = cycle(full_list)
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='{0} (AUC = {1:0.2f})'
                 ''.format(association[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right", prop={"size":9})
    #fig.savefig('temp.png', dpi=fig.dpi)

    print('AUC macro average :' + str(roc_auc["macro"]))
    print('AUC micro average :' + str(roc_auc["micro"]))


    return fig

