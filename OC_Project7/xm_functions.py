# visual libs

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


def count_nan(df, feature_name):
    print(df[feature_name].isna().sum(), "null/nan values for:", feature_name)


def fill_feature_with_mean(df, feature_name):
    count_nan(feature_name)
    print("Filling null", feature_name, "with mean:", round(df[feature_name].mean(), 2))
    df[feature_name] = df[feature_name].fillna(round(df[feature_name].mean(), 2))


def fill_feature_with_empty(df, feature_name):
    count_nan(feature_name)
    if df.dtypes[feature_name] in ["int64", "float64"]:
        print("Filling null", feature_name, "with zeros")
        df[feature_name] = df[feature_name].fillna(0)
    elif df.dtypes[feature_name] in ["object"]:
        print("Filling null", feature_name, "with empty string")
        df[feature_name] = df[feature_name].fillna("")
    else:
        print("error, unmanaged pandas dtype")


def fill_feature_with_value(df, feature_name, value):
    count_nan(feature_name)
    print("Filling null", feature_name, "with mean:", value)
    df[feature_name] = df[feature_name].fillna(value)


def myround(x, base):
    return int(base * round(float(x) / base))

def np_str(np_array):
    return str(' '.join(map(str, np_array)))

def plot_rep_stats(
    df,
    x_column,
    y_column,
    x_column_label="default",
    y_column_label="default",
    show_distrib=False,
    ci=None,
    pltshow=True,
):
    if x_column_label == "default":
        x_column_label = x_column
    if y_column_label == "default":
        y_column_label = y_column

    x_ordered = df[[x_column, y_column]].groupby(x_column).mean().sort_values(y_column)
    uniques = df[x_column].unique()
    uniques.sort()

    if show_distrib == True:
        fig = plt.hist(df[x_column], bins=len(uniques))
        silent = plt.title("Repartition per " + x_column_label)
        silent = plt.ylabel("Count")
        silent = plt.xlabel(x_column_label)
        plt.show()

    if len(uniques) < 25:
        fig = sns.boxplot(x_column, y_column, data=df)
        fig = sns.pointplot(
            uniques, df.groupby([x_column])[y_column].mean().values, join=True
        )
        silent = fig.set_title(y_column_label + " par " + x_column_label)
        silent = fig.set(xlabel=x_column_label, ylabel=y_column_label)
    else:
        # fig = sns.pointplot(x_ordered.index.values, x_ordered[y_column].values, join=True, order = x_ordered.index.values)
        fig = sns.lineplot(
            x_column,
            y_column,
            data=df,
            style_order=x_ordered.index.values,
            ci=ci,
            legend="full",
        )
        silent = plt.title(y_column_label + " par " + x_column_label)
        silent = plt.ylabel(y_column_label)
        silent = plt.xlabel(x_column_label)
    if pltshow:
        plt.show()
    return fig


def display_plot(
    cv_scores, cv_scores_std, alpha_space, param_name="", model_name="hyperparam"
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(
        alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2
    )

    ax.set_title(model_name + " " + param_name + " tuning")
    ax.set_ylabel("CV Score +/- Std Error")
    ax.set_xlabel("Hyperparam " + param_name)
    ax.axhline(np.max(cv_scores), linestyle="--", color=".5")
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    # log good for logspaces
    # ax.set_xscale('log')
    plt.show()


def hyper_param(
    model, X, y, param_name="default", linespace_multiplier=1, best_score_alpha=1
):
    if linespace_multiplier > 10:
        print("10 Loops, aborting.")
        return
    model_name = model.__class__.__name__
    if param_name == "default":
        if model_name == "KNeighborsClassifier":
            param_name = "n_neighbors"
        elif model_name == "RandomForestClassifier":
            param_name = "n_estimators"
        else:
            param_name = "alpha"

    # Note : should switch to param_name
    if param_name in ["min_samples_split", "max_leaf_nodes"]:
        alpha_space = (
            np.linspace(2, 10 * 2 ^ linespace_multiplier, 10).round(0).astype(int)
        )
    elif param_name in ["n_estimators"]:
        alpha_space = (
            np.linspace(2, 20 * 2 ^ linespace_multiplier, 10).round(0).astype(int)
        )
    elif param_name in ["learning_rate"]:
        alpha_space = np.linspace(0.01, 0.1 * linespace_multiplier, 10)
    elif model_name in ["KNeighborsClassifier", "RandomForestClassifier", "DecisionTreeClassifier"]:
        # min_hyper = 1+(15*linespace_multiplier -15)/4
        min_hyper = 1
        alpha_space = (
            np.linspace(min_hyper, 15 * 2 ^ linespace_multiplier, 10)
            .round(0)
            .astype(int)
        )
    else:
        alpha_space = np.logspace(-10 * linespace_multiplier, 0, 10)
    scores = []
    scores_std = []
    model.normalize = True

    # Compute scores over range of alphas
    for alpha in alpha_space:
        setattr(model, param_name, alpha)
        # k folds
        cv_scores = cross_val_score(model, X, y, cv=3)
        scores.append(np.mean(cv_scores))
        scores_std.append(np.std(cv_scores))

    max_score = max(scores)
    returned_alpha = alpha_space[scores.index(max_score)]
    print("Highest score is", max_score, "Hyperparam ", param_name, ":", returned_alpha)

    continue_alpha_search = False
    print(returned_alpha)
    if model_name in ["KNeighborsClassifier", "RandomForestClassifier", "DecisionTreeClassifier"]:
        if best_score_alpha < returned_alpha:
            continue_alpha_search = True
    else:
        if best_score_alpha > returned_alpha:
            continue_alpha_search = True

    if continue_alpha_search:
        return hyper_param(
            model,
            X,
            y,
            param_name,
            linespace_multiplier + 1,
            best_score_alpha=returned_alpha,
        )

    plt.rcParams["figure.figsize"] = (10, 4)
    display_plot(scores, scores_std, alpha_space, param_name, model_name)
    return returned_alpha



def legend(plt, title=None, xlabel=None, ylabel=None):
    if plt.__class__.__name__ == 'AxesSubplot':
        if title:
            silent = plt.set_title(title)
        if xlabel:
            silent = plt.set_xlabel(xlabel)
        if ylabel:
            silent = plt.set_ylabel(ylabel)
    else:
        if title:
            silent = plt.title(title)
        if xlabel:
            silent = plt.xlabel(xlabel)
        if ylabel:
            silent = plt.ylabel(ylabel)

def trace_mean(plt, dist_series):
    mean_val = dist_series.mean()
    plt.axvline(mean_val, color='k', linestyle='dashed', linewidth=1)
    
    if plt.__class__.__name__ == 'AxesSubplot':
        _, max_ = plt.get_ylim()
    plt.text(mean_val) -110, max_+max_ , 'Mean  {:.2f}'.format(mean_val)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                         text = True
                         ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(shrink = 0.7)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if text:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.tight_layout()

def fig_size(width=20, height = 4, font = 18):
    plt.rcParams["figure.figsize"] = (width,height)
    plt.rcParams["font.size"] = (font)
    
def plt_roc(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    silent = plt.title('Receiver Operating Characteristic')
    silent = plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    silent = plt.legend(loc = 'lower right')
    silent = plt.plot([0, 1], [0, 1],'r--')
    silent = plt.xlim([0, 1])
    silent = plt.ylim([0, 1])
    silent = plt.ylabel('True Positive Rate')
    silent = plt.xlabel('False Positive Rate')
    silent = plt.show()
    
def imshow(img):
    if len(img.shape) ==2:
        plt.imshow(img, cmap='gray')
    else :
        plt.imshow(cv2.cvtColor(img, cv2.CV_32S))
    plt.show()