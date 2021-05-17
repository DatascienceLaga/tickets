# TICKETS DEMAND PROJECT - POC
___
## Context
This project is a complete study aiming to predict the number of demands and reservations that can be booked according 
to a trajectory (origin-destination) and the time difference between reservation and the actual departure of the train. 
For this first POC, a number of `Machine Learning` and `Deep Learning` algorithms were tested and compared. During this 
process, other tasks, like hyper-parameters tuning were worth testing in order to show their 
utility in optimizing and improving the models' performances.

## Table of contents
1. Complete study
    1. Reading the data
    2. Preprocessing
    3. Data analysis
    4. Data split
    5. Training the models
    6. Tuning the models
    7. Evaluating the models
2. Code automation
3. Docker image

In each of these sections, we will give a detailed description of how the experiences were held and what were hypotheses
 that were made for each step.
___
## Complete study
In this part, we are diving into the details of the results listed in the notebook. So we will be explaining our choices
, experiences and how to interpret the results.

### Reading the data
In order to read the data, we used a custom function based enabling to decompress the .lz4 format file, transform the 
data from bytes to string and finally format the data into a dataframe. This function was applied for both train and test
datasets. Usually, we use the read_csv method from pandas to read compressed csv files. Although, the lz4 format isn't 
taken into account by the function. 

### Preprocessing
The received data was actually well processed and treated and there wasn't much to change, except the fact that we have 
to encode the categorical variables, especially origin and destination stations, into unique binary variables. This 
action is called one hot encoded vectors. This was the only additional preprocessing step that needed to be added in our
pipeline to train our models. Plus, for evaluation issues, we added a supplementary column the categorises the 
`sale_day_x` variable into intervals.

### Data analysis
In the notebook, there is a whole section that shows some graphics about some variables from our dataset. The goal is to
comprehend the data and highlight the main characteristics of the dataset. The insights of this section were mainly 
about the proportions of train-test data (here: 95% train vs 5% test). Furthermore, the training data were records of 
sales of a historical interval of 1 year and a half and the test set contains data of about two weeks following the 
training set dates. We also have got some statistics about the target variable, as to know `demand`. So, based on origin
, destination and sale_day_x, we get demands varying from zero to 143 but 75% of the observations have more or less 3 
demands. In addition to that, other graphs are given in the notebook with the corresponding comments.

### Data split
As explained before, the train and test data were given in two separate folders. So the splitting part was already 
specified and we only had to recreate the input and output matrices. So, we ended up with a total of 4 matrices (X_train
, X_test, y_train, y_test) where the input matrices contain the selected variables for training (in our cases all variables
were selected to train) and the output arrays contain only the target variable.

### Training the models
For our study, we selected a number of models to test and to compare. From each family of models we selected two types 
of models. From the machine learning algorithms, we train the `Random forest` and the `XGBoost` models. On the other 
hand, we train a `DNN model` as well as an `RNN model`. All of these models were trained on basic parameters at first 
then were tuned. Plus, for the neural architectures, we had to tweak some hyper-parameters manually in order to find 
simple yet performant models such as the number of layers and the number of nodes for each layer.

### Tuning the models
As mentioned previously, each model was trained with its default parameters then we used a grid search on some of the 
most relevant parameters in order to find the best set of parameters to increase the performances of the model. Indeed, 
by using this method of optimisation, we could reach a difference of 5% in terms of R2 score, which is a high increase 
due to this optimisation algorithm. In the conducted experiments, each time we used a grid-search to optimize the choice
of the model's parameters, we observed a slight increase of scores and as cited above, the highest increase was around 
5% of the R2 score.

### Evaluating the models
In this part, we actually return an elaborate report that contains global and aggregated scores, such as the `MAE`, 
`MSE`, `relative error` and the `R2 score`. We start by showing scores that were aggregated on origin-destination and 
the corresponding interval of sale_day_x. Based on these aggregated values, we compute the cited above errors then go 
through another level of aggregation that takes only origin and destination stations into account then compute the same 
errors. Finally, on a global level we assess those scores on the full plain test set and plot the prediction as well as 
the real values for visual comparison. 
>**PS: to evaluate the prediction, we deleted all of the negative values that the model could generate by taking he 
> maximum between the prediction and the value 0. Also, since the target variable is an integer type and our regression 
> models render float numbers, we rounded up the predictions to the closet integer value to try to have the best 
> presentation of the prediction.**

>**Another important note is that especially for the relative error, its formula includes dividing by the predicted 
> values which means that in some cases we will receive `inf` as result meaning that in that case the prediction was 
> equal to zero and we had to divide a number by zero. On the other hand, dividing zero by another zero (real values and
> predictions are both equal to zero) would render `NaN` as relative error.**

Furthermore, after evaluating our models, we can have an idea about the most relevant features that the model trained on
and used to create its predictions. For both, RF and XGBoost models, variables like `sale_day_x` and travel time were 
very important to the model, which actually confirms the correlation between these variables, and the target variable 
shown in the heatmap that figures on the notebook. 

## Code automation
In order to industrialize this solution, we had to automate the code by creating different scripts, each one having a 
specific role in the whole pipeline. To do so, 5 folders were created, namely: `data`, `features`, `utils`, `training` 
and `evaluate`. Each one of these folders contains a script or more that has functions used for the corresponding task. 
These scripts contain mainly the same functions used in the notebook. Thus, a `main.py` script is put on the same level 
as these folders in order to import their functions and run them in the correct order. After running the main script, 
the steps of the pipeline would show in order to know which part is being executed. Also, the training and the 
evaluation results will show u in the console.

>**The main function takes as arguments: `The path that contains the folder of the training and the test data` as well 
> as `a boolean argument that is True if we want to run the grid-search, False otherwise`.**

>**Note: In this case the scores tables would appear on the console and three interactive html graphs would pop-up on 
> your browser where each graph corresponds to results of each one of the models. They have the specificity of zooming, 
> auto-scaling and showing interactive values** 

In the quickstart section, we will see how to run the main script for local trials.

## Docker image
We are also using a docker image of this package in order to run it without having dependencies issues, where the main 
script is exposed in order to run it the same way as we do for local use.
___
# Quickstart
Here, we show how to run each of the codes in order to run the experiments and get the corresponding results.

First, other than using the docker image, we will need to respect the dependencies for this project. Therefore, we will
need to install the requirements as listed in the `requirements.txt` file. But, before that, a good practice for doing 
so is to create a virtual environment specific for this project where we will install these packages:

      #Create the virtual env
      python -m venv my-env

      #Activate the virtual env
      ##On windows
      my-env\Scripts\activate.bat
      ##On Unix/MacOS
      source my-env/bin/activate

      #Install requirements
      python -m pip install requirements.txt

Next, depending on which support to try, there is a different approach to run:
## Jupyter notebook
In order to run this type of files, you need to install anaconda then run the `jupyter-notebook` or the `jupyter-lab` 
then access to the folder that contains the notebook and launch it. Since it was already launched previously, you will 
find previous results in the outputs of each cell, but you can run it as well in and try the code on your own.

## Local `main.py` launch
In that case, run the terminal inside the project folder on the same level where the `notebooks` and `src` are located
then run:

      
      python src/main.py <path-to-data> <True if use grid-search, False otherwise>
      #Default example
      python src/main.py "dataset" False