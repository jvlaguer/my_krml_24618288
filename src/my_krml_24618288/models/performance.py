# Solution:
def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import root_mean_squared_error as rmse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {rmse(y_actuals, y_preds)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")

def display_classifier_scores(y_actuals, y_preds):
    """Evaluating accuracy, precision, recall, and F1

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target

    Returns
    -------
    Display everything onto a dataframe
    """

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    dict_eval_metrics = {'accuracy': accuracy_score,
                 'precision': precision_score,
                 'recall': recall_score,
                 'f1': f1_score,
                }

    list_scores = [scorer(y_actuals, y_preds) for scorer in dict_eval_metrics.values()]
    display(pd.DataFrame(list_scores, index=dict_eval_metrics.keys(), columns=['Results']))

def plot_confusion_matrix(y_actuals, y_preds):
    """Displaying confusion matrix

    Parameters
    ----------
    y_actuals : Numpy Array
        Actual target
    y_preds : Numpy Array
        Predicted target 

    Returns
    -------
    """

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from matplotlib import pyplot as plt

    cm = confusion_matrix(y_actuals, y_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def plot_auroc_curve(y_actuals, y_probs):
    """Displaying AUROC Curve

    Parameters
    ----------
    y_actuals : Numpy Array
        Actual target
    y_probs : Numpy Array
        Predicted target probabilities

    Returns
    -------
    """

    from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
    from matplotlib import pyplot as plt
    
    print('AUROC score: {:.4f}'.format(roc_auc_score(y_actuals, y_probs)))

    fpr, tpr, _ = roc_curve(y_actuals, y_probs)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5,5)
    plt.plot([0, 1], [0.5, 0.5], color = 'g', linestyle='--')
    plt.show()