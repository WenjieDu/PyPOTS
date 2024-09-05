from pypots.classification import CSAI
import pickle
from pypots.optim import Adam
from pypots.utils.metrics import calc_mae
import torch
from pypots.utils.metrics import calc_binary_classification_metrics

kfold_data = pickle.load(open(r"C:\Users\jsphr\Documents\Imputation\data\data_nan.pkl", 'rb'))
kfold_label = pickle.load(open(r"C:\Users\jsphr\Documents\Imputation\data\label.pkl", 'rb'))


train_data = kfold_data[0][0]
train_label = kfold_label[0][0]

valid_data = kfold_data[0][1]
valid_label = kfold_label[0][1]


test_data = kfold_data[0][2]
test_label = kfold_label[0][2]

dataset_for_training = {
    "X": train_data,
    "y": train_label
}

dataset_for_validating = {
    "X": valid_data,
    "y": valid_label
}

dataset_for_testing = {
    "X": test_data,
    "y": test_label
}

csai = CSAI(n_steps = 48,
            n_features = 35,
            rnn_hidden_size = 108,
            imputation_weight = 0.3,
            consistency_weight = 0.1,
            classification_weight = 1,
            n_classes = 1,
            removal_percent = 10,
            increase_factor = 0.1,
            compute_intervals = True,
            step_channels = 512,
            batch_size = 32, 
            epochs = 10, 
            patience = None, 
            optimizer = Adam(lr=1e-3),
            num_workers = 0, 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            saving_path = r'C:\Users\jsphr\Documents\Imputation', 
            model_saving_strategy = "best", 
            verbose = True)

csai.fit(dataset_for_training, dataset_for_validating)
csai_results = csai.predict(dataset_for_testing)
print(csai_results.keys())
classification_predictions = csai_results['classification']
metrics = calc_binary_classification_metrics(classification_predictions, dataset_for_testing["y"])

# or0 = csai_results["X_ori"]
# indi = csai_results["indicating_mask"]

# mae = calc_mae(imputed_data, or0, indi)
# print(mae)

print("Testing classification metrics: \n"
    f'ROC_AUC: {metrics["roc_auc"]}, \n'
    f'PR_AUC: {metrics["pr_auc"]},\n'
    f'F1: {metrics["f1"]},\n'
    f'Precision: {metrics["precision"]},\n'
    f'Recall: {metrics["recall"]},\n'
)