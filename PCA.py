

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import seaborn as sns
from numpy import cov
from numpy.linalg import eig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
def plot_airfoil(x, y, figsize=(12,4)):
    """Plot the airfoil desribed by nodes (x[i], y[i]) """

    plt.figure(figsize=figsize)
    plt.plot(x,y)

    ax = plt.gca()
    ax.set_ylim([-0.1, 0.1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")  
    ax.set_title("Airfoil")

    plt.show()

def plot_cp_distribution(x, cp, title="Cp Distribution", figsize=(8,6)):
    """Plot the Cp distribution of one simulation
    
    Don't worry if you have never seen a Cp distribution. You can just treat it as an dimensionless output vector of size 192"""

    plt.figure(figsize=figsize)
    plt.plot(x, cp, "o-")

    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1]) # invert the y-axis of the plot
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Cp")    
    ax.set_title(title)

    plt.show()


def main():
    # load input data
    flow_conditions = pd.read_csv("flow_conditions.csv", index_col=0)
    print("flow_conditions:")
    print(flow_conditions)
    
    # load simulation results
    surface_flow_sim_results = pd.read_csv("surface_flow_sim_results.csv", index_col=0)
    print("surface_flow_sim_results:")    
    print(surface_flow_sim_results)

    # get the coordinates of the airfoil mesh nodes
    # (the geometry does not change between simulation runs, so the first 192 values are enough)
    nodes_x = surface_flow_sim_results["x"][:192].values
    nodes_y = surface_flow_sim_results["y"][:192].values

    #plot_airfoil(nodes_x, nodes_y)

    # the only output variable we care about in this assignment is the pressure coefficient Cp
    # we can extract the values and turn the pandas.Series object into a numpy array (2D matrix of size n_sim x n_nodes)
    Cp = surface_flow_sim_results["Pressure_Coefficient"].values
    Cp = Cp.reshape(-1,192)
    print("Cp.shape:", Cp.shape)
    Y = Cp # this is the output matrix from the assignment
    #standardizing before PCA
    scaler = StandardScaler()
    scaler.fit(Y)
    trans = scaler.transform(Y)
    #Calculating the covariance matrix
    covariance_matrix = np.cov(trans.T)
    #Eigendecomposition of the Covariance Matrix
    eigen_values_cmp, eigen_vectors_cmp = np.linalg.eig(covariance_matrix)
    eigen_values = eigen_values_cmp.real
    eigen_vectors = eigen_vectors_cmp.real
    #Determining how many principal components to be used
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i / sum(eigen_values)) * 100)


    #
    print(variance_explained)
    cumulative_variance_explained = np.cumsum(variance_explained)
    print(cumulative_variance_explained)
    a_list = list(range(1, 193))
    sns.lineplot(x=a_list[:8], y=cumulative_variance_explained[:8])
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Explained variance vs Number of components")
    plt.show()

    projection_matrix = (eigen_vectors.T[:][:8]).T
    # print(projection_matrix)
    Y_pca = Y.dot(projection_matrix)
    print(Y_pca)




    # the same can be done with the input data (flow conditions)
    X = flow_conditions[["Ma", "AoA"]].values
    print("X.shape:", X.shape)


    # now do the analysis...

    #train regression model
    X_train, X_val, y_train, y_val = train_test_split(X, Y_pca, random_state=1, test_size=0.2)
    regr = MLPRegressor(hidden_layer_sizes=(512, 512, 512), activation="tanh", random_state=1, max_iter=2000,
                        early_stopping=True).fit(X_train, y_train)

    y_pred_val = regr.predict(X_val)
    #mean squared error
    print("MSE= ", (mean_squared_error(y_pred_val, y_val)))

    flow_conditions_test = pd.read_csv("flow_conditions_test.csv", index_col=0)
    X_test = flow_conditions_test[["Ma", "AoA"]].values
    loss = regr.loss_curve_
    iter = regr.n_iter_
    a_list = list(range(0, iter))
    plt.plot(a_list, loss, label="loss_curve")
    plt.xlabel("Number of iterations")
    plt.ylabel("loss")
    plt.title("Regression Model loss curve")
    plt.legend()
    plt.show()
    # prediction
    Y_pred_test = regr.predict(X_test)
    Y_pred = Y_pred_test.dot(projection_matrix.T)
    Y_pred = scaler.inverse_transform(Y_pred)


    plot_cp_distribution(nodes_x, Y_pred[0],
                         title="Cp test Distribution i=1 at $M_{\infty}$=" + f"{X_test[0, 0]} and AoA={X_test[0, 1]} [deg.]")
    
    Cp_output_data = Y_pred

    pd.DataFrame(Cp_output_data).to_csv("cp_output_data.csv")




if __name__ == "__main__":
    main()
