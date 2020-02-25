# Example 001

This example gives a framework for using a python script to train a
RandomForest on a dataset and write some statistics of that trained
model to disk.

We will do this while sweeping over a suite of datasets and possible
parameters.


The run this example on MARCC the folowing steps must be completed:

```
git clone https://github.com/neurodata/marcc_examples.git
ml python/3.8
python -m venv ~/env_examples
pip install -r marcc_examples/requirements.txt
```


## The files involved

### [`skrf.py`](./skrf.py)

This is the workhorse python script.  It takes command line arguments as
parameters to tell the script which dataset to use with which
parameters.  


### [`workerScript.scr`](./workerScript.scr)

This is the slurm scirpt that sets up the job and cluster parameters
such as number of nodes per job, cpus, memory, ...


### [`cleanUp.scr`](./cleanUp.scr)

This is a slurm script that runs the clean up scripts which could be
anything from moving files around, aggregating outputs from the previous
jobs, plotting, etc.

### [`plot.py`](./plot.py)

This is an example clean up script in python for aggregating and
plotting the results.

### [`config.dat`](./config.dat)

This is a file that specifies other parameters that are fixed accross
each individual job.  There are two sections, `default` for MARCC and
`dev` for use on my local box.  This helps with setting parameters such
as number of runs, or number of trees to an acceptable level for
testing before release on the cluster.

### [`runParameters.dat`](./runParameters.dat)

This is a space delimited file the contains the parameters used for each
individual job.  Specifically, it has `dataID` `runID`.  The parameters
on line `i` are used for job `i` in teh job array.

### [`MASTER.scr`](./MASTER.scr)

This is the MASTER script.  It is in charge of submitting the main job
as a job array with the appropriate lenght and submitting the cleanUp
job as a dependancy.


---


## Running on MARCC



