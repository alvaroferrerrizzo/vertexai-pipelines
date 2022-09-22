@component(
   packages_to_install=["pandas", "scikit-learn", "google-cloud-aiplatform"]
)
def train(in_experiment_name:str, 
          in_experiment_training_set: str,
          in_vertexai_region: str, 
          in_vertexai_projectid: str, 
          in_csv_path: InputPath('CSV_DATASET'), 
          model_type: str, 
          saved_model: Output[Model]
         ):
    
    import pandas as pd  
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics
    from google.cloud import aiplatform
    from datetime import datetime
    import joblib
    import os
    import random
    idn = random.randint(0,1000)


    dataset = pd.read_csv(in_csv_path)
    
    print(dataset.head())
    
    X = dataset.loc[:, dataset.columns != 'Churn']
    Y = dataset['Churn']
    
    print(X.columns.values)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
    
    aiplatform.init(
       project=in_vertexai_projectid,
       location=in_vertexai_region,
       experiment=in_experiment_name
    )
    
    run_id = f"run-{idn}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    aiplatform.start_run(run_id)
    
    #Choose which model to train
    if model_type == 'svm':
        from sklearn import svm
        model = svm.LinearSVC()
        
    elif model_type == 'random_forrest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
        
    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        
    model.fit(X_train, Y_train)
    
    saved_model_path = os.path.join(saved_model.path.replace("saved_model",""), 'model.joblib')
    joblib.dump(model, saved_model_path)

     
    predicted = model.predict(X_test)
    
    print("accuracy: {}".format(metrics.accuracy_score(Y_test, predicted)))
    print("f1 score macro: {}".format(metrics.f1_score(Y_test, predicted, average='macro')   )  )
    print("f1 score micro: {}".format(metrics.f1_score(Y_test, predicted, average='micro') ))
    print("precision score: {}".format(metrics.precision_score(Y_test, predicted, average='macro') ))
    print("recall score: {}".format(metrics.recall_score(Y_test, predicted, average='macro') ))
    print("hamming_loss: {}".format(metrics.hamming_loss(Y_test, predicted)))
    print("log_loss: {}".format(metrics.log_loss(Y_test, predicted)))
    print("zero_one_loss: {}".format(metrics.zero_one_loss(Y_test, predicted)))
    print("AUC&ROC: {}".format(metrics.roc_auc_score(Y_test, predicted)))
    print("matthews_corrcoef: {}".format(metrics.matthews_corrcoef(Y_test, predicted) ))
    
    
    training_params = {
        'training_set': in_experiment_training_set,
        'model_type': model_type,
        'dataset_path': in_csv_path,
        'model_path': saved_model_path
    }
    
    training_metrics = {
        'model_accuracy': metrics.accuracy_score(Y_test, predicted),
        'model_precision': metrics.precision_score(Y_test, predicted, average='macro'),
        'model_recall': metrics.recall_score(Y_test, predicted, average='macro'),
        'model_logloss': metrics.log_loss(Y_test, predicted),
        'model_auc_roc': metrics.roc_auc_score(Y_test, predicted)
    }
    
    aiplatform.log_params(training_params)
    aiplatform.log_metrics(training_metrics)
