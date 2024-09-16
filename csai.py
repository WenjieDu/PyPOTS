import pickle
import logging
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, 
    balanced_accuracy_score, recall_score, precision_score, f1_score, 
    PrecisionRecallDisplay, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
)
from pypots.classification import CSAI as classify
from pypots.imputation import CSAI as imputer
from pypots.optim import Adam
from pypots.utils.metrics import calc_mae, calc_mre, calc_binary_classification_metrics
import json
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_performance_brits(y, y_score, y_pred, pr_display=False, cm_display=False, roc_display=False):

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_score)
    prec_macro = precision_score(y, y_pred, average='macro')
    recall_macro = recall_score(y, y_pred, average='macro')
    f1_macro = f1_score(y, y_pred, average='macro')
    bal_acc = balanced_accuracy_score(y, y_pred)

    if pr_display:
        prec, recall, _ = precision_recall_curve(y, y_score)
        PrecisionRecallDisplay(precision=prec, recall=recall).plot()

    if cm_display:
        cm = confusion_matrix(y, y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    if roc_display:
        fpr, tpr, _ = roc_curve(y, y_score)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

    return acc, auc, prec_macro, recall_macro, f1_macro, bal_acc


def load_data(kfold_data_path, kfold_label_path, fold=2):

    kfold_data = pickle.load(open(kfold_data_path, 'rb'))
    kfold_label = pickle.load(open(kfold_label_path, 'rb'))

    return {
        "train": (kfold_data[fold][0], kfold_label[fold][0]),
        "valid": (kfold_data[fold][1], kfold_label[fold][1]),
        "test": (kfold_data[fold][2], kfold_label[fold][2])
    }


def get_csai_model(mode, **kwargs):

    if mode == 'classify':
        return classify(**kwargs)
    else:
        return imputer(**kwargs)


def run_pypots(mode, data):
    
    
    dataset_for_training = {"X": data["train"][0], "y": data["train"][1]}
    dataset_for_validating = {"X": data["valid"][0], "y": data["valid"][1]}
    dataset_for_testing = {"X": data["test"][0], "y": data["test"][1]}

    model_params = {
        "n_steps": 48,
        "n_features": 35,
        "rnn_hidden_size": 108,
        "imputation_weight": 0.3,
        "consistency_weight": 0.1,
        "removal_percent": 10,
        "increase_factor": 0.1,
        "compute_intervals": True,
        "step_channels": 512,
        "batch_size": 64,
        "epochs": 200,
        "patience": 5,
        "optimizer": Adam(lr=0.0005, weight_decay=0.00001),
        "num_workers": 0,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "saving_path": '/scratch/users/k23031260/PyPOTS',
        "model_saving_strategy": "best",
        "verbose": True
    }

    if mode == 'classify':
        model_params["classification_weight"] = 1
        model_params["n_classes"] = 1

    csai_model = get_csai_model(mode, **model_params)
    
    logger.info("\n Training..................................... \n")
    csai_model.fit(dataset_for_training, dataset_for_validating)
    
    logger.info("\n Testing...................................... \n")
    csai_results = csai_model.predict(dataset_for_testing)
    
    if mode == 'classify':
        classification_predictions = csai_results['classification']
        metrics = calc_binary_classification_metrics(classification_predictions, dataset_for_testing["y"])
        
        y_pred = np.round(classification_predictions)
        acc, auc, prec_macro, recall_macro, f1_macro, bal_acc = calculate_performance_brits(
            dataset_for_testing["y"], classification_predictions, y_pred
        )

        return {
            "PyPots_evaluation": {
                "ROC_AUC": metrics['roc_auc'],
                "PR_AUC": metrics['pr_auc'],
                "F1": metrics['f1'],
                "Precision": metrics['precision'],
                "Recall": metrics['recall']
            },
            "CSAI_evaluation": {
                "ACC": acc,
                "AUC": auc,
                "Prec_Macro": prec_macro,
                "Recall_Macro": recall_macro,
                "F1_Macro": f1_macro,
                "Bal_ACC": bal_acc
        }
        }

    else:

        original_data = csai_results["X_ori"]
        indicating_mask = csai_results["indicating_mask"]
        imputed_data = csai_results['imputation']

        mae = calc_mae(imputed_data, original_data, indicating_mask)
        mre = calc_mre(imputed_data, original_data, indicating_mask)

        return {
            "MAE": mae,
            "MRE": mre
        }



if __name__ == "__main__":

    mode = 'impute'  
    data_path = "/scratch/users/k23031260/data/physionet/data_nan.pkl"
    label_path = "/scratch/users/k23031260/data/physionet/label.pkl"

    all_fold_results = []
    for fold in range(5):
        logger.info(f"\n Fold {fold} started ...................................... \n")
        data = load_data(data_path, label_path, fold)
        results = run_pypots(mode, data)
        all_fold_results.append(results)
        logger.info(f"\n Fold {fold} ended ........................................ \n")
        logger.info(f"\n Fold {fold} Results: \n{json.dumps(results, indent=4)} \n")

    logger.info(f"\n Average performance across all folds.......................... \n")
    if mode == 'impute':
        avg_result = {metric: np.mean([fold[metric] for fold in all_fold_results]) for metric in all_fold_results[0].keys()}
        logger.info(f"\n Imputation performance: \n {avg_result} \n")
    else:
        PyPots_evaluation = [{k: v for k,v in fold_result['PyPots_evaluation'].items()} for fold_result in all_fold_results]
        CSAI_evaluation = [{k: v for k,v in fold_result['CSAI_evaluation'].items()} for fold_result in all_fold_results]

        avg_result = {metric: np.mean([fold[metric] for fold in PyPots_evaluation]) for metric in PyPots_evaluation[0].keys()}
        logger.info(f"\n Classification performance using Pypots evaluation: \n {avg_result} \n")

        avg_result = {metric: np.mean([fold[metric] for fold in CSAI_evaluation]) for metric in CSAI_evaluation[0].keys()}
        logger.info(f"\n Classification performance using CSAI evaluation: \n {avg_result} \n")


