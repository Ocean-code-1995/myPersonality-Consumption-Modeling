import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns
from sklearn import metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Any
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, mean_squared_error
from math import ceil

# enable to see all columns:
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')



def check_common_ids(df1: pd.DataFrame, id_col_df1: str, df2: pd.DataFrame, id_col_df2: str) -> None:
    '''
    Returns boolean value in case there are common ids in both data frames.
    as well as the amount of identical ids.
    '''
    # ensure that the dataype of both ids are the same (string)
    df1[id_col_df1] = df1[id_col_df1].astype(str)
    df2[id_col_df2] = df2[id_col_df2].astype(str)

    # check for common ids, and count them
    common_ids_bool = df1[id_col_df1].isin(df2[id_col_df2])
    any_common      = common_ids_bool.any()
    amount_common   = sum(common_ids_bool)

    print(f"The two dfs have common ids: {any_common}")
    print(f"Number of common ids: {amount_common:_}")

def display_data_shape(df: pd.DataFrame) -> None:
    '''
    Displays the shape of the data frame.
    '''
    n_observaitons = f" - Number of observations: {df.shape[0]:,}"
    n_columns      = f" - Number of columns: {df.shape[1]:,}"
    print(f"Data dimensions:\n{69*'-'}")
    print(n_observaitons)
    print(n_columns)
    

def plot_continuous_features_target(df: pd.DataFrame, features: List[str], target: str, title: str) -> None:
    """Plotting continuous features against the target variable as stripplot. Median is also displayed."""
    ncols = 6
    nplots = len(features)
    nrows = ceil(nplots / ncols)

    # Define the size of each individual subplot
    width_per_subplot = 3   # Example width in inches
    height_per_subplot = 4  # Example height in inches

    # Calculate total figure size
    figsize = (ncols * width_per_subplot, nrows * height_per_subplot)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    axes = axes.ravel()

    # Other parts of the function remain the same
    for n, feature in enumerate(features):
        sns.stripplot(data=df, x=target, y=feature, hue=target, palette={True: 'red', False: 'blue'},
                    jitter=0.3, size=9, linewidth=0.75, alpha=0.5, ax=axes[n])


        sns.boxplot(data=df, x=target, y=feature, showmeans=False,
                    meanprops={'color': 'darkgreen', 'ls': '-', 'lw': 2},
                    medianprops={'color': 'lightgreen', 'ls': '-', 'lw': 6},
                    showfliers=False, showcaps=False, showbox=False,
                    whiskerprops={'visible': False}, ax=axes[n], zorder=10)
        # discrad legend
        axes[n].get_legend().remove()

        axes[n].set_xticks([0, 1])
        axes[n].set_xticklabels(['False', 'True'], fontsize=11, rotation=0)
        axes[n].set_title(feature, fontsize=14, weight='bold')
        axes[n].set_xlabel('Label', fontsize=12, weight='bold')
        axes[n].set_ylabel('LIWC Expression', fontsize=11)

    # Adjust layout for a better fit
    fig.suptitle(title, fontsize=22, weight='bold').set_y(1.001)
    plt.tight_layout(pad=3.0)

# Statistical Tests:
#----------------------------------------------------------------------------------------------
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as mt
import pingouin as pg
from typing import List


def welch_anova(data: pd.DataFrame, features: List[str], target: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Determines significant features with respect to a binary target using Welch's ANOVA.
    Returns the features with p-values less than the specified alpha level.
    """

    significant_features = []
    significant_p_values = []

    for feature in features:
        # Perform Welch's ANOVA
        aov = pg.welch_anova(dv=feature, between=target, data=data)
        p_value = aov.at[0, 'p-unc']  # Extract the p-value

        if p_value < alpha:
            significant_features.append(feature)
            significant_p_values.append(round(p_value, 4))

    print(f"{len(significant_features)} out of {len(features)} features are significant with respect to {target} based on Welch's ANOVA p-values:")

    return pd.DataFrame(
        {
            'feature': significant_features,
            'p_value': significant_p_values,
        }
    ).sort_values(by='p_value', ascending=True).reset_index(drop=True)

# Example usage:
# welch_results = welch_anova(data=my_dataframe, features=my_feature_list, target='my_binary_target')


def mannwhitney(data: pd.DataFrame, features: List[str], target: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Determines significant features with respect to a binary target using the Mann-Whitney U test.
    Returns the features with p-values less than the specified alpha level.
    """

    significant_features = []
    significant_p_values = []

    for feature in features:
        p_value = mannwhitneyu(
                            x = data.query(f"{target} == True")[feature],
                            y = data.query(f"{target} == False")[feature]
        )[1]

        if p_value < alpha:
            significant_features.append(feature)
            significant_p_values.append(round(p_value, 4))

    print(f"{len(significant_features)} out of {len(features)} features are significant with respect to {target} based on raw p-values:")

    return pd.DataFrame(
        {
            'feature': significant_features,
            'p_value': significant_p_values,
        }
    ).sort_values(by='p_value', ascending=True).reset_index(drop=True)


def mannwhitney_adjusted(data: pd.DataFrame, features: List[str], target: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Determines significant features with respect to a binary target using the Mann-Whitney U test.
    It also adjusts the p-values using the Benjamini-Hochberg procedure because we are testing multiple hypotheses,
    where alpha accumulation is a problem ("family-wise error rate").
    """

    mannwhitney_features = []
    raw_p_values = []

    for feature in features:
        p_value = mannwhitneyu(
                            x = data[data[target] == True][feature],
                            y = data[data[target] == False][feature]
        )[1]
        mannwhitney_features.append(feature)
        raw_p_values.append(p_value)

    # Adjust the p-values using Benjamini-Hochberg procedure
    adjusted_p_values = mt.multipletests(raw_p_values, method='fdr_bh')[1]

    # Combine features with their adjusted p-values
    feature_p_values = zip(mannwhitney_features, adjusted_p_values)

    # Filter significant features after adjustment
    significant_results = [(feature, round(p, 4)) for feature, p in feature_p_values if p < alpha]

    print(f"{len(significant_results)} out of {len(features)} features are significant with respect to {target} after p-value adjustment:")

    # Convert to DataFrame and sort
    return pd.DataFrame(significant_results, columns=['feature', 'adjusted_p_value']).sort_values(by='adjusted_p_value', ascending=True).reset_index(drop=True)



def plot_target_ratios(df: pd.DataFrame, targets: list) -> None:
    """Plot the percentage of each target variable in a stacked bar chart.
    """
    # Create a subplot figure with len(targets) rows and 1 column
    fig = make_subplots(rows=len(targets), cols=1, shared_xaxes=True)

    # Colors for the bars
    colors = ['midnightblue', 'red']

    for idx, target in enumerate(targets):
        # Calculate the percentage of the target variable
        true_count = df[target].sum()
        false_count = len(df) - true_count
        true_percentage = round((true_count / len(df)) * 100, 2)
        false_percentage = round((false_count / len(df)) * 100, 2)

        # Add False bar
        fig.add_trace(
            go.Bar(
                y=[target],
                x=[false_percentage],
                name='False',
                text=f'{false_percentage}%',
                textposition='inside',
                orientation='h',
                marker_color=colors[0],
                hoverinfo='name+x',
                showlegend=(idx == 0)  # Only show the legend for the first set
            ), row=idx+1, col=1
        )

        # Add True bar
        fig.add_trace(
            go.Bar(
                y=[target],
                x=[true_percentage],
                name='True',
                text=f'{true_percentage}%',
                textposition='inside',
                orientation='h',
                marker_color=colors[1],
                hoverinfo='name+x',
                showlegend=(idx == 0)  # Only show the legend for the first set
            ), row=idx+1, col=1
        )

    # Update layout for a stacked bar chart
    fig.update_layout(
        barmode='stack',
        title='Stacked Bar Chart of Binary Target Variables',
        legend=dict(
            traceorder='normal',
            itemsizing='constant'
        ),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    # Set the x-axis title for the last plot only
    fig.update_xaxes(title_text="Percentage", row=len(targets), col=1, tickformat=',.2f')

    # Set the y-axis ticks to display the target names
    for i, target in enumerate(targets, 1):
        fig.update_yaxes(tickvals=[target], ticktext=[target], row=i, col=1)

    # Show the figure
    fig.show()


def classification_reports(
        model: BaseEstimator,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        threshold: float = 0.5, title_suffix: str = ''
    ) -> None:
    """
    Generate and display a classification reports for the training and test sets.
    """

    # Make predictions on the dataset using predict_proba
    y_pred_train_proba = model.predict_proba(X_train)
    y_pred_train = (y_pred_train_proba[:, 1] >= threshold).astype(int)

    y_pred_test_proba = model.predict_proba(X_test)
    y_pred_test = (y_pred_test_proba[:, 1] >= threshold).astype(int)

    print(f"Classification Reports {title_suffix}:\n{66*'_'}")
    # Classification Report for training set
    print(f"\n- Train Set:\n{59*'~'}")
    report = metrics.classification_report(y_train, y_pred_train, output_dict=True)
    df = pd.DataFrame(report).transpose()
    display(df.style.background_gradient(
        cmap='Oranges', 
        vmin=0.0, vmax=1.0, 
        subset=pd.IndexSlice[['0', '1', 'accuracy'], ['precision', 'recall', 'f1-score']]
    ))

    # Classification Report for training set
    print(f"\n- Test Set:\n{59*'~'}")
    report = metrics.classification_report(y_test, y_pred_test, output_dict=True)
    df = pd.DataFrame(report).transpose()
    display(df.style.background_gradient(
        cmap='Oranges',
        vmin=0.0, vmax=1.0,
        subset=pd.IndexSlice[['0', '1', 'accuracy'], ['precision', 'recall', 'f1-score']]
    ))


def classification_evaluation_report(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, threshold: float=0.5, figsize: tuple=(11,4), title_suffix: str = '') -> None:
    """
    Generate and display a classification report including a confusion matrix, a horizontal bar chart for the confusion matrix,
    and a classification report.

    Parameters:
    model (classifier): The trained classifier model.
    X (DataFrame): Feature set for making predictions.
    y (Series): True target values.
    threshold (float): Threshold for converting predicted probabilities to binary predictions.
    title_suffix (str): Suffix for the title in plots to differentiate between train/test sets.
    """

    # Make predictions on the dataset using predict_proba
    y_pred_proba = model.predict_proba(X)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    TN, FP, FN, TP = confusion_matrix.ravel()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)

    # Plotting the Confusion Matrix
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot(ax=ax1, cmap='plasma', values_format='d')
    ax1.set_ylabel('Actual Label', size=11)
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels(ax1.get_yticklabels(), size=10)
    ax1.set_xlabel('Predicted Label', size=11)
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(ax1.get_xticklabels(), size=10)
    ax1.set_title(f'Confusion Matrix {title_suffix}', fontsize=13, fontweight='bold')
    ax1.grid(False)

    # Horizontal Bar Chart for Confusion Matrix

    plot_data = pd.DataFrame({"0": [TN, FN], "1": [FP, TP]}, index=["0", "1"])
    # Reversing the order of rows for the bar chart
    plot_data = plot_data.iloc[::-1]

    plot_data.plot(kind="barh", stacked=True, ax=ax2, color=["midnightblue", "red"], fontsize=12, edgecolor='black', linewidth=1)
    ax2.set_title(f'Classification Error {title_suffix}', fontsize=13, fontweight='bold')
    ax2.set_xlabel("# Predictions", fontsize=11)
    ax2.set_ylabel("Actual Label", fontsize=11)
    ax2.set_yticklabels(ax2.get_yticklabels(), size=10)
    ax2.legend(title='Predicted Label', title_fontsize=9, loc='center', frameon=True, framealpha=1, facecolor='aliceblue', edgecolor='red', ncol=3)

    plt.tight_layout()
    plt.show()

    # Printing Class Distribution
    print(f"Class Distribution in Dataset:\n{59*'~'}")
    zero_count = y.value_counts()[0]
    one_count  = y.value_counts()[1]
    zero_perc = round(y.value_counts(normalize=True)[0], 2) * 100
    one_perc  = round(y.value_counts(normalize=True)[1], 2) * 100
    print(f"- Class 0: {zero_count} ({zero_perc}%)\n- Class 1: {one_count} ({one_perc}%)")

    # Classification Report
    print(f"\nClassification Report {title_suffix}:\n{59*'~'}")
    report = metrics.classification_report(y, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    display(df.style.background_gradient(
        cmap='Oranges', 
        vmin=0.0, vmax=1.0, 
        subset=pd.IndexSlice[['0', '1', 'accuracy'], ['precision', 'recall', 'f1-score']]
    ))


from sklearn.metrics import precision_score, recall_score, f1_score


def threshold_analysis_barplot(
                model: BaseEstimator, X: pd.DataFrame, y: pd.Series, 
                thresholds: List[float] = np.arange(0.20, 0.8, 0.05).round(2),
                figsize: tuple = (14, 6)
    ) -> None:
    """
    Plots precision, recall, and F1 score for both classes of a binary target at various thresholds.

    Parameters:
    - model (BaseEstimator): Trained binary classification model.
    - X (pd.DataFrame): Feature set for making predictions.
    - y (pd.Series): True target labels.
    - thresholds (List[float]): List of thresholds to evaluate.
    """
    
    # 1) create data for plotting:
    y_prob = model.predict_proba(X)
    adj_threshold = pd.DataFrame()
    cols = ['Metrics']

    for threshold in thresholds:
        y_pred_1 = (y_prob[:, 1] > threshold).astype(int)
        y_pred_0 = 1 - y_pred_1

        precision_1 = round(precision_score(y, y_pred_1, pos_label=1), 3)
        recall_1    = round(recall_score(y, y_pred_1, pos_label=1), 3)
        f1_1        = round(f1_score(y, y_pred_1, pos_label=1), 3)

        precision_0 = round(precision_score(y, y_pred_0, pos_label=0), 3)
        recall_0    = round(recall_score(y, y_pred_0, pos_label=0), 3)
        f1_0        = round(f1_score(y, y_pred_0, pos_label=0), 3)

        name = f'Threshold: {threshold}'
        adj_threshold[name] = [precision_1, recall_1, f1_1, precision_0, recall_0, f1_0]
        cols.append(name)

    adj_threshold.index = ['Precision_1', 'Recall_1', 'F1_1', 'Precision_0', 'Recall_0', 'F1_0']
    adj_threshold = adj_threshold.reset_index()
    adj_threshold.columns = cols


    class_1 = adj_threshold[adj_threshold['Metrics'].isin(['Precision_1', 'Recall_1', 'F1_1'])]
    class_0 = adj_threshold[adj_threshold['Metrics'].isin(['Precision_0', 'Recall_0', 'F1_0'])]

    # 2) plot data:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    fig.suptitle('Metrics across Varying Thresholds', fontsize=20, fontweight='bold')

    c1 = class_1.plot(x='Metrics', kind='bar', stacked=False, cmap='twilight', ax=ax[0])
    c1.legend(loc='center left', bbox_to_anchor=(1, 0.02))
    c1.set_xticklabels(labels=['Precision', 'Recall', 'F1'], rotation=0)
    c1.set_xlabel('')
    c1.set_ylabel('Class 1 (%)', fontsize=15, fontweight='bold')

    c0 = class_0.plot(x='Metrics', kind='bar', stacked=False, cmap='twilight', ax=ax[1])
    c0.legend('')
    c0.set_xticklabels(labels=['Precision', 'Recall', 'F1'], rotation=0)
    c0.set_xlabel('')
    c0.set_ylabel('Class 0 (%)', fontsize=15, fontweight='bold')

def threshold_analysis_lineplot(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, num_thresholds=50, figsize=(14, 6)) -> None:
    # https://towardsdatascience.com/stop-using-0-5-as-the-threshold-for-your-binary-classifier-8d7168290d44
    y_scores = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0, 1, num_thresholds)
    data = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        y_pred_0 = (y_scores < threshold).astype(int)

        data.append({
            'Threshold': threshold,
            'Precision_Class_1': precision_score(y, y_pred, zero_division=0),
            'Recall_Class_1': recall_score(y, y_pred, zero_division=0),
            'F1_Class_1': f1_score(y, y_pred, zero_division=0),
            'Flagged_Class_1': np.sum(y_pred),
            'Precision_Class_0': precision_score(y, y_pred_0, pos_label=0, zero_division=0),
            'Recall_Class_0': recall_score(y, y_pred_0, pos_label=0, zero_division=0),
            'F1_Class_0': f1_score(y, y_pred_0, pos_label=0, zero_division=0),
            'Flagged_Class_0': np.sum(y_pred_0)
        })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Metrics across Varying Thresholds', fontsize=16, fontweight='bold')


    # Plotting for Class 1
    sns.lineplot(data=df, x='Threshold', y='Precision_Class_1', label='Precision', color='blue', ax=axes[0])
    sns.lineplot(data=df, x='Threshold', y='Recall_Class_1', label='Recall', color='green', ax=axes[0])
    sns.lineplot(data=df, x='Threshold', y='F1_Class_1', label='F1 Score', color='orange', ax=axes[0])
    ax2 = axes[0].twinx()
    sns.lineplot(data=df, x='Threshold', y='Flagged_Class_1', label='Flagged Instances', color='red', ax=ax2)
    axes[0].set_title('Class 1')
    axes[0].set_ylabel('Metric Value')
    axes[0].legend('')
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', colors='tab:red')

    # Plotting for Class 0
    sns.lineplot(data=df, x='Threshold', y='Precision_Class_0', label='Precision', color='blue', ax=axes[1])
    sns.lineplot(data=df, x='Threshold', y='Recall_Class_0', label='Recall', color='green', ax=axes[1])
    sns.lineplot(data=df, x='Threshold', y='F1_Class_0', label='F1 Score', color='orange', ax=axes[1])
    ax4 = axes[1].twinx()

    sns.lineplot(data=df, x='Threshold', y='Flagged_Class_0', label='Flagged Instances', color='red', ax=ax4)
    axes[1].set_title('Class 0')
    axes[1].set_ylabel('Metric Value')
    axes[1].legend('')
    ax4.set_ylabel('')
    ax4.tick_params(axis='y', colors='tab:red')

    # Create custom legend
    blue_line = mlines.Line2D([], [], color='blue', label='Precision')
    green_line = mlines.Line2D([], [], color='green', label='Recall')
    orange_line = mlines.Line2D([], [], color='orange', label='F1 Score')
    fig.legend(handles=[blue_line, green_line, orange_line], loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=4, shadow=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.show()

def feature_importance_plot(model: BaseEstimator, feature_names: list, figsize:tuple=(8, 10), horizontal:bool=True) -> None:
    """
    Generate and display a feature importance plot based on a fitted model.

    Parameters:
    model: The trained classifier model that provides feature importances.
    feature_names (list): The list of names for the features used in the model.
    """

    # Get the feature importance score from the fitted classifier
    feature_importance = model.feature_importances_

    # Create a dataframe to store the results
    results = pd.DataFrame({
                        'Feature'   : feature_names,
                        'Importance': feature_importance
    })

    # Sort the dataframe by importance
    results = results.sort_values('Importance', ascending=False).reset_index(drop=True)

    if not horizontal:
        # Plot the dataframe as a horizontal bar chart
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', hue='Importance', data=results, palette='coolwarm_r', edgecolor='lightgreen', width=0.2)
        plt.title('Feature Importance Scores', fontsize=18, fontweight='bold', y=1.05)
        plt.xlabel('Score', fontsize=15)
        plt.ylabel('Feature Space', fontsize=15)
        plt.show()
    else:
        # Plot the dataframe as a vertical bar chart
        plt.figure(figsize=figsize)
        sns.barplot(x='Feature', y='Importance', data=results, palette='coolwarm_r', edgecolor='lightgreen')
        plt.title('Feature Importance Scores', fontsize=18, fontweight='bold', y=1.05)
        plt.xlabel('Feature Space', fontsize=15)
        plt.ylabel('Score', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        plt.show()


def regressor_evaluator(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame.style:
    """
    Evaluates a regression model on the train and test sets and returns a styled DataFrame.

    Parameters:
        model: A fitted regression model.
        X_train: Training features.
        y_train: Training target variable.
        X_test: Testing features.
        y_test: Testing target variable.

    Returns:
        A styled DataFrame with R2 score and RMSE for both training and testing sets.
    """
    # Metrics calculation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Store in DataFrame
    metrics = pd.DataFrame({'Train': [r2_train, rmse_train], 'Test': [r2_test, rmse_test]},
                           index=['R2 Score', 'RMSE'])


    return metrics.style.background_gradient(cmap='Reds', axis=1, vmin=0, vmax=1)



import shap
def global_shap_plots_tree(model, X, figsize, max_display=10):
    """Displaying global SHAP plots for a tree-based model.

    Args:
        model: A tree-based model.
        X: The train / test data.
        figsize: The size of the figure.
    """
    # Your existing code for creating SHAP values
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)
    shap_values_expl = shap.Explanation(shap_values, feature_names=X.columns)

    # Set up the matplotlib figure and axes
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Global SHAP Plots', weight='bold', size=25).set_y(1.01)

    # Plot SHAP values for test data as a bar plot on the first subplot
    plt.sca(ax[0])  # Set the current axes to the first subplot
    shap.plots.bar(shap_values_expl, max_display=max_display, show=False)
    ax[0].set_title("Bar Plot", weight='bold', size=18)

    # Plot SHAP values for test data as a beeswarm plot on the second subplot
    plt.sca(ax[1])  # Set the current axes to the second subplot
    shap.summary_plot(shap_values, X, plot_type='dot', show=False, max_display=max_display, plot_size=None)
    ax[1].set_title("Beeswarm Plot", weight='bold', size=18)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()