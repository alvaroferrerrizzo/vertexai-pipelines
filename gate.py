from typing import NamedTuple

@component(
   packages_to_install=["pandas", "google-cloud-aiplatform"]
)
def gate(in_experiment_name: str,
         in_experiment_training_set: str,
         in_vertexai_region: str,
         in_vertexai_projectid: str,
         model1: Input[Model],
         model2: Input[Model],
         model3: Input[Model]
        )-> NamedTuple(
           'winner_output',
            [
                ('experiment_info', str),
                ('is_current_champion', bool)
            ]
        ):
    
    from google.cloud import aiplatform
    import json
    from collections import namedtuple
    
    aiplatform.init(
       project=      in_vertexai_projectid,
       location=     in_vertexai_region,
       experiment =  in_experiment_name
    )
    
    ## get vertex AI model object corresponding to <champion model> from ModelRegistry - use labels: experiment_name 
    champion_model = None
    champion_model_exists = False
    
    model_filter_str='labels.experiment_name="'+in_experiment_name+'"'
    print("Model filter string: "+model_filter_str)
    
    models = aiplatform.Model.list(
        filter=model_filter_str
    )
    
    if len(models)>0:
        champion_model_exists = True
        champion_model = models[0]
        print(champion_model.display_name)
        champion_model_experiment_run_id = champion_model.labels['experiment_run_id']
    
    
    ## fetch experiment run details for current <training set>:
    experiment_df = aiplatform.get_experiment_df()
    experiment_df = experiment_df[experiment_df.experiment_name == in_experiment_name]
    
    
    challengers_experiment_run_info =  experiment_df[experiment_df["param.training_set"] == in_experiment_training_set]
    
    print("Challengers:")
    print(challengers_experiment_run_info.to_string())
    
    if champion_model != None:
       current_champion_experiment_run_info = experiment_df[experiment_df["run_name"] == champion_model_experiment_run_id]
    
    decision_metric_name = "metric.model_auc_roc"
    
    ### fetch best experiment_run_id from challengers
    best_challenger_experiment_run_info = challengers_experiment_run_info[
        challengers_experiment_run_info[decision_metric_name]==challengers_experiment_run_info[decision_metric_name].max()
    ]
    
    print("Best challenger")
    print(best_challenger_experiment_run_info.to_string())
    
    winner_experiment_run_info = None
    
    winner_is_current_champion = False
    if champion_model != None: 
        winner_experiment_run_info = current_champion_experiment_run_info
        winner_is_current_champion = True
        
        ## Final: best_challenger vs champion
        if best_challenger_experiment_run_info[decision_metric_name].values[0]>current_champion_experiment_run_info[decision_metric_name].values[0]:
            ## best challenger is the new winner
            winner_experiment_run_info = best_challenger_experiment_run_info
            winner_is_current_champion = False
    else: 
        winner_experiment_run_info = best_challenger_experiment_run_info
        winner_is_current_champion = False
    
    winner_experiment_info = {
           "experiment_name": winner_experiment_run_info["experiment_name"].values[0],
           "experiment_run_id": winner_experiment_run_info["run_name"].values[0]
    }
    
    print("winner:")
    print(winner_experiment_info)
    
    ##https://www.kubeflow.org/docs/components/pipelines/sdk/python-function-components/#pass-by-file
    winner_namedtuple = namedtuple('winner_output', ['experiment_info', 'is_current_champion'])
    
    return winner_namedtuple(json.dumps(winner_experiment_info), winner_is_current_champion)
