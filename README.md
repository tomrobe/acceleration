# README
## Usage

This package may be installed, but it is not necessary. Basic usage is possible by simply creating a configuration file and a run script, as detailed below. Together, they can be used to produce whatever desired scenario.

### Configuration file

Inside the configuration directory are sample configuration files (in particular, `dummy_sweep.json`). To create your own, simply copy it and adjust the values.
In order to sweep over one or more values, you need to pick the `"sweep"` `"sweep_type"` and specify the values in the `"sweep_config"` as shown. Note that this will override the corresponding values in `"base_parameters"` and `"initial_conditions"`.
Inside `"sweep_config"`, two groups are offered:
1. `"parameters"`: These give a cartesian product of the parameters, in the sense that the program will sweep over every combination of values.
2. `"paired_parameters"`: All lists here are zipped together, so the sweep will use the first of each, then the second of each, and so on. Note that everything here needs to be the same length.
For instance, if there is one list in `"parameters"` of length 3, and 2 lists in `"paired_parameters"` of length 4, the total amount of solutions will be 3*4=12.

### Script

An example run script is provided in the scripts directory as `test.py`. It does two(-ish) things:
1. Solve the simulation based on the specified configuration file. You need to specify the path relative to the base directory.
2. Plot the results. You will need to look into `src/acceleration/plotting.py` to see the available functions, or just write your own here.
The first plot call (`plot_subplots()`) in the test.py is a WIP general function, which you can ignore for now. It will hopefully replace the need for all the others eventually and shift the plot config more towards the run script. It is also the reason for importing parts of the analysis into this script, which is otherwise handled in the plot calls.
3. You can write your own analysis here as well, as well as print any desired output values. Just remember that the solutions object is the collection of all solutions and check `src/acceleration/models` for what the solution objects consist of.

Then simply run `python path/to/script.py`.
