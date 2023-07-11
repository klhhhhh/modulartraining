from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python sortParasMeta.py'
experiment.config.trial_code_directory = '.'
search_space = {
    'lr': {'_type': 'loguniform', '_value': [1e-6, 1e-1]},
    'weight_decay': {'_type': 'loguniform', '_value': [1e-3, 1e-1]},
}
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 50
experiment.config.trial_concurrency = 2
# max_experiment_duration = '1h'
# experiment.run(8082)
experiment.resume ( "yksd9xb5" , port = 8082 , wait_completion = True , debug = False )
input('Press enter to quit')
experiment.stop()
