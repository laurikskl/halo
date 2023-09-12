# setup

`brew install conda`

`conda env create -f environment.yml`

`conda activate halo`

Select the environment in vscode from the bottom right corner.

# adding new packages

`conda install <package_name>`

Add to `dependencies.yml` with package version included

# updating your environment

`conda env update -f environment.yml`

# if something is fucked

`conda deactivate`
`conda remove -n halo --all`

Do setup again
 