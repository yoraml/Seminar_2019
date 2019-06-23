import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


if __name__ == '__main__':

    df = pd.read_csv('outputs/task_e6d1e108-2630-4b1b-826b-b399d2faac89.csv')
    df['AUC_score'] = df['AUC_score'].astype(float)
    table = df[['Outer_iteration_number', 'number_of_features', 'best_feature_selection_model', 'AUC_score']].round(2).T
    df = df[['Outer_iteration_number', 'AUC_score']]
    df = df.sort_values('AUC_score')
    X=df['AUC_score']
    mu = X.mean()
    std = X.std()
    title = "AUC Distribution classification results: mu = %.2f,  std = %.2f" % (mu, std)

    plt.hist(X, bins=4, density=True, alpha=0.6, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=1)
    plt.xlabel('AUC Score')
    plt.title(title)
    plt.show()

    a=1
