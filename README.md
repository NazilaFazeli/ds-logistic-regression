[![Shipping files](https://github.com/neuefische/ds-logistic-regression/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-logistic-regression/actions/workflows/workflow-02.yml)

# Logistic Regression

In this repository, you will learn about logistic regression.

## The way to success:

Please work together as **Pair-Programmers** through all the notebooks
in this particular order:

1. [Logistic Regression](1_Logistic_Regression.ipynb)
2. [Logistic Regression sklearn](2_Logistic_Regression_sklearn.ipynb)
3. [Multiclass Classification](3_Multiclass_Classification.ipynb)
4. [Logistic Regression Exercise](4_Logistic_Regression_Exercise.ipynb)

The first two notebooks will show you how logistic regression works graphically
and with scikit-learn.
In the third notebook, you will find an example of a multiclass classification using one of the most popular data science data sets.
In the fourth notebook, it's your time to implement logistic regression on a new data set. If you get stuck you can have a look at the solution notebook in the solution branch. 

## Set up your Environment

Please make sure you have forked the repo and set up a new virtual environment. For this purpose you can use the following commands:

The added [requirements file](requirements.txt) contains all libraries and dependencies we need to execute the Logistic Regression notebooks.

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file. M1 shizzle.*

### **`macOS`** type the following commands : 


- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.12.7
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.12.7
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
    ```
    pyenv local 3.12.7
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

The data used in this notebook is saved in a `.zip` file. To unzip it, copy the block below into your terminal:

```Bash
unzip data.zip
```

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file.*
