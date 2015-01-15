This will hold the main script for the experiment.

./run_cartpole.py  <parameter_file> <parameter1=value> ... <parametern=value>

The parameter file is a python script containing key=value pairs describing what you wish to run,
and how you wish to run it. Examples can be found under ../params

You may with to override parameters on the command line, for this you can include key=value
command line parameters. This allows you to perform runs that modify particular
hyperparameters while keeping everything else constant based on a parameter file.
