# Helpful Tips


### your `.bashrc`

These are some helpful additions to your `~/.bashrc` file.
```
alias rm='rm -i'
alias sidle='sinfo | grep idle'
alias smix='sinfo | grep mix'
```



### `pip install`

When using python I suggest making sure you order your steps thusly,

1. Load the desired version of python with `ml python/<V.v>`.
2. Create a virtual environment (in your home directory seems safest).
3. Activate your environment.
4. Make sure you are using the correct pip with `which pip`. 
5. Use `pip install` with the `--no-cache-dir` option to install your
   packages.
