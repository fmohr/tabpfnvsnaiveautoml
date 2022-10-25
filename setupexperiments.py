from py_experimenter.experimenter import PyExperimenter

experimenter = PyExperimenter(config_file="config/experiments.cfg")
#experimenter.reset_experiments("error")
#experimenter.reset_experiments("running")
experimenter.fill_table_from_config()