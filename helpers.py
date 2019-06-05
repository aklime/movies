import numpy as np
from pprint import pprint
from joblib import dump, load
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import cm

from sklearn.metrics import r2_score, median_absolute_error

# plots style
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

#ocena uzyskanych wyników -> błąd względny w postaci: (|oczekiwana_odpowiedź - odpowiedź_systemu| / oczekiwana_odpowiedź)*100% dla każdej próbki, a następnie uśrednić wyniki dla wszystkich n próbek. I na jego podstawie optymalizować parametry.
def evaluate(y_test, predictions):
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    return accuracy

def scores(y_test, predictions):
    mae =  metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    r2 =  metrics.r2_score(y_test, predictions)
    accuracy = evaluate(y_test, predictions)
    
	
    scorings = {'mae': mae, 
     'mse': mse,
     'rmse': rmse,
     'r2': r2,
     'accuracy': accuracy}
    
    #pprint(scorings)
    
    return scorings

def get_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def plot_error_dist(y_test, predictions, prefix, model):
    fig = plt.figure()
    sns.distplot((y_test - predictions),bins=50, color="#74A608")
    plt.xlabel('')
    plt.title('Rozkład błędu')
    plt.savefig('images/' + prefix + '/error_distribution_' + model +'.jpg', dpi = 300)
    

def plot_residuals_vs_fitted(y_test, predictions, prefix, model):
    plot = plt.figure()
    plot.axes[0] = sns.residplot(predictions, y_test,
                          lowess=True,
                          scatter_kws={'s':5, 'alpha': 0.8, 'color': "#74A608" },
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot.axes[0].set_title('')
    plot.axes[0].set_xlabel('Predykcje')
    plot.axes[0].set_ylabel('Reszty');
    plt.savefig('images/' + prefix + '/residuals_vs_fitted_' + model + '.jpg', dpi = 300)

def plot_true_vs_pred(y_test, predictions, prefix, model):
    plt.figure()
    plt.xlim(1,10)
    plt.ylim(1,10)
    plt.scatter(y_test, predictions, s=5,color="#74A608")
    plt.plot([1, 10], [1, 10], '--k', color = "#1E4006")
    plt.text(1.5, 9, r'$R^2$=%.2f, MAE=%.2f' % (
    r2_score(y_test, predictions), median_absolute_error(y_test, predictions)))
    plt.xlabel('Średnie oceny filmów')
    plt.ylabel('Predykcje')
    plt.title(model)
    plt.savefig('images/'+ prefix+ '/true_vs_pred_' + model +'.jpg', dpi = 300)


def plot(y_test, predictions, prefix, name):
    # plots style
    sns.set(font_scale=1)
    sns.set_style('whitegrid')
    plot_error_dist(y_test, predictions, prefix, name)
    plot_residuals_vs_fitted(y_test, predictions, prefix, name)
    plot_true_vs_pred(y_test, predictions, prefix, name)

def open_models(path, name):
    model = load(path + name + '.joblib')
    model_std = load(path + name + '_std.joblib')
    model_mm = load(path + name + '_mm.joblib')
    
    models = {
        name : model,
        name + '_std': model_std,
        name + '_mm': model_mm,
    }

    return models


def open_std(path, names):
    models = {}
    for name in names: 
        model_std = load(path + name + '_std.joblib')
        models[name+'_std'] = model_std
        
    return models

def open_mm(path, names):
    models = {}
    for name in names: 
        model_mm = load(path + name + '_mm.joblib')
        models[name+'_mm'] = model_mm
        
    return models

def open_original(path, names):
    models = {}
    for name in names: 
        model = load(path + name + '.joblib')
        models[name] = model
        
    return models


def autolabel(ax, rects, rot=0, s=12, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.55, 'left': 0.45}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}%'.format(height), ha=ha[xpos], va='bottom', rotation=rot, size=s)


        