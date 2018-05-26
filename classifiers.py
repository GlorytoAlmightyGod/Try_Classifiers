"""

Following are some of the key points worth mentioning related to this problem :

    Attached program analyzes these models one by one -> Linear regression, Polynomial regression, SVM, Random Forests.

    One could easily increase the analyzed models by just adding an appropriate new line for corresponding model in the code.

    Random Forests is the best fitting model for provided data out of all the models mentioned above !

    Results for each model are printed on the terminal for models in the same order mentioned in the first point.

"""





import pandas as pd
import matplotlib.pyplot as plt                                 # for plotting

from sklearn.metrics import mean_squared_error                  # for RMS calculations
from math import sqrt

from sklearn import linear_model                                # for linear regression classifier

from sklearn import svm                                         # for support vector machine classifier
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor              # for Random forests

from sklearn.preprocessing import PolynomialFeatures            # for Quadratic or higher degree regression models
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


file = 'training_data.xlsx'   # Assign spreadsheet filename to `file`

xl = pd.ExcelFile(file)         # Load spreadsheet

df = xl.parse('Sheet1')         # Load a sheet into a DataFrame by name: df and make a 2D array arr with all important data
arr = df.values

y_out = arr[:,0]            # assign first coloumn as final function value
x_in = arr[:,1:]            # assign rest of the table as input parameter to our classifier function

classifiers = [             # list of all classifiers that we want to analyze

    linear_model.LinearRegression(),
    make_pipeline(PolynomialFeatures(3), Ridge()),
    svm.SVR(),
    RandomForestRegressor(n_estimators = 50, random_state = 0)

]

print("\n  Abs Errors Sum         RMS Error         Max Abs Error")

i = 1                       # an iterator used to change linewidth in graph to distinguish classifiers

for item in classifiers:                            # looping over all classifiers
    
    clf = item                                      # creating object of concerned model
    clf.fit(x_in, y_out)                            # Train the model
    
    y_pred = clf.predict(x_in)                      # Make predictions
    
    diff = abs(y_out - y_pred)                      # Compute absolute error for each row of a given classifier   
    rms = sqrt(mean_squared_error(y_out, y_pred))   # Evaluate rms for each classifier
    
    print(sum(diff),rms,max(diff))                  # print & plot stuff
    plt.plot(range(y_out.size), diff, linewidth = i)
    i = i + 1                                       # increment iterator
    
plt.show()              # plot the graph!



