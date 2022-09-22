from kfp.dsl import pipeline
from kfp.dsl import Condition

@pipeline(name="wf-churn")
def pipeline(
    in_bigquery_projectid: str = 'defaultprojectid',
    in_bigquery_dataset: str = 'telcosandbox',
    in_corr_threshold: float = 0.05,
    in_experiment_name: str = "telcochurn",
    in_experiment_training_set: str = "telcochurn",
    in_vertexai_projectid: str = "",
    in_vertexai_region: str = "",
    in_vertex_serving_machine_type: str = "n1-standard-4",
    in_vertex_serving_min_replicas: int = 1,
    in_vertex_serving_max_replicas: int = 2
    
):
    
    import json
    
    #### STEP1: STAGING
    staging_task = stage(in_bigquery_projectid,
                         in_bigquery_projectid,
                         in_bigquery_dataset
                        )
    
    ### STEP2: FEATURE ENGINEERING
    feature_eng_task = preprocess(staging_task.output, 
                                  in_bigquery_projectid,
                                  in_bigquery_projectid, 
                                  in_bigquery_dataset, 
                                  in_experiment_name, 
                                  in_experiment_training_set,
                                  in_corr_threshold)
    
    
    ### STEP3: TRAIN CHALLENGERS
    train_task_svm =            train(in_experiment_name, 
                                      in_experiment_training_set, 
                                      in_vertexai_region, 
                                      in_vertexai_projectid, 
                                      feature_eng_task.output, 
                                      'svm',)
    train_task_random_forrest = train(in_experiment_name,
                                      in_experiment_training_set,
                                      in_vertexai_region, 
                                      in_vertexai_projectid, 
                                      feature_eng_task.output, 
                                      'random_forrest')
    train_task_decision_tree = train(in_experiment_name, 
                                      in_experiment_training_set, 
                                      in_vertexai_region, 
                                      in_vertexai_projectid, 
                                      feature_eng_task.output, 
                                      'decision_tree')
    
    
    #### STEP4: GATE - Identify best challenger and compare with current champion
    evaluation_gate_task = gate(in_experiment_name, 
                                in_experiment_training_set, 
                                in_vertexai_region, 
                                in_vertexai_projectid, 
                                train_task_svm.output, 
                                train_task_random_forrest.output, 
                                train_task_decision_tree.output)
     
    
    with Condition(
        evaluation_gate_task.outputs['is_current_champion'] == "false", name="deploy_new_champion"
    ): 
        
        ### STEP 5&6 Register new Chamption and deploy it to endpoint
        result = deploy(in_experiment_name, 
                        in_experiment_training_set, 
                        in_vertexai_region, 
                        in_vertexai_projectid,
                        evaluation_gate_task.outputs['experiment_info'],
                        in_vertex_serving_machine_type,
                        in_vertex_serving_min_replicas,
                        in_vertex_serving_max_replicas
                       )
