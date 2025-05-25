# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
df = pd.read_csv("cardio_train.csv", sep=';')
df.head ()

# %% [markdown]
# 
# **Inspect data for Null, Columns, Duplicates, and Basic statistics of data distribution**

# %%
df.columns

# %%
df.shape

# %%
df.info()

# %%
df.isna().sum()

# %%
df.describe()

# %% [markdown]
# **INSIGHTS FROM THE DESCRIPTION ABOVE**
# * Minimum values of systolic and diastolic bp shouldn't be negative
# * Maximum valus shouln't be up to 16020 and 11000 respectively

# %%
df.duplicated().sum()

# %%
df.nunique()

# %% [markdown]
# Features:
# 
# Age | Objective Feature | age | int (days) Height | Objective Feature | height | int (cm) | Weight | Objective Feature | weight | float (kg) | Gender | Objective Feature | gender | categorical code | Systolic blood pressure | Examination Feature | ap_hi | int | Diastolic blood pressure | Examination Feature | ap_lo | int | Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal | Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal | Smoking | Subjective Feature | smoke | binary | Alcohol intake | Subjective Feature | alco | binary | Physical activity | Subjective Feature | active | binary | Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
# 
# Refer to df.head() above for better interpretation

# %%
df['age_years'] = df['age']/365
# Then use y='age_years' in violinplot

# %%
# import visaulization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# using histogram and KDE

# %%

# Create a 1x2 subplot grid
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# Plot 1: Histogram of weight (green)
sns.histplot(data=df, x='weight', bins=30, kde=True, color='b', ax=axs[0])
axs[0].set_title('Distribution of Weight')

# Plot 2: Weight distribution by cardio class (hue)
sns.histplot(data=df, x='weight', kde=True, hue='cardio', ax=axs[1])
axs[1].set_title('Weight Distribution by Cardio Status')

plt.tight_layout()  # Prevents overlapping labels
plt.show()

# %%
plt.figure(figsize=(25, 6))  # Set overall figure size

# Subplot 1: All Genders
plt.subplot(1, 3, 1)  # 1 row, 3 cols, position 1
sns.histplot(data=df, x='weight', kde=True, hue='cardio', palette='viridis')
plt.title('Weight Distribution (All Genders)')

# Subplot 2: Gender = 1 (Men)
plt.subplot(1, 3, 2)  # Position 2
sns.histplot(data=df[df['gender'] == 1], x='weight', kde=True, hue='cardio', palette='coolwarm')
plt.title('Weight Distribution (Men)')

# Subplot 3: Gender = 2 (Women)
plt.subplot(1, 3, 3)  # Position 3
sns.histplot(data=df[df['gender'] == 2], x='weight', kde=True, hue='cardio', palette='magma')
plt.title('Weight Distribution (Women)')

plt.tight_layout()  # Prevent overlapping
plt.show()

# %%
# 1. Create a 1x2 subplot grid and set figure size
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# 2. First subplot: Height distribution (all patients)
sns.histplot(data=df, x='height', bins=30, kde=True, color='g', ax=axs[0])
axs[0].set_title('Height Distribution (All Patients)')
axs[0].set_xlabel('Height (cm)')

# 3. Second subplot: Height distribution by cardio status
sns.histplot(data=df, x='height', kde=True, hue='cardio', ax=axs[1])
axs[0].set_title('Height Distribution by Cardio Status')
axs[1].set_xlabel('Height (cm)')
axs[1].legend(title='Cardio', labels=['No Disease', 'Disease'])

# 4. Adjust layout and display
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insights**
# 
# * Higher weight is associated with cardiovascular diseases, and simalr for all genders
# * Height does not influence incidence of cardiovascular diseases

# %%

#Converting age in days to years
age_years = df['age']/365
# Create figure and subplot grid (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(15, 7))

# First subplot: Age distribution (all patients)
sns.histplot(data=df, x='age_years', bins=30, kde=True, color='g', ax=axs[0])
axs[0].set_title('Age Distribution (All Patients)')
axs[0].set_xlabel('Age (years)')

# Second subplot: Age distribution by cholesterol level
sns.histplot(data=df, x='age_years', kde=True, hue='cholesterol', palette='viridis', ax=axs[1])
axs[1].set_title('Age Distribution by Cholesterol Level')
axs[1].set_xlabel('Age (years)')
axs[1].legend(title='Cholesterol', 
             labels=['Well Above Normal', 'Above Normal', 'Normal'])

# Adjust layout and display
plt.tight_layout()
plt.show()

# %%
# Set up the figure and subplots
plt.figure(figsize=(18, 8))
#Converting age in days to years
age_years = df['age']/365

# 1. Height Violin Plot
plt.subplot(1, 4, 1)
sns.violinplot(y='height', data=df, color='red', linewidth=3, inner='quartile')
plt.title('Height Distribution', fontsize=12, pad=20)
plt.ylabel('Height (cm)', fontsize=10)

# 2. Weight Violin Plot
plt.subplot(1, 4, 2)
sns.violinplot(y='weight', data=df, color='green', linewidth=3, inner='quartile')
plt.title('Weight Distribution', fontsize=12, pad=20)
plt.ylabel('Weight (kg)', fontsize=10)

# 3. Age Violin Plot
plt.subplot(1, 4, 3)
sns.violinplot(y='age_years', data=df, color='blue', linewidth=3, inner='quartile')
plt.title('Age Distribution', fontsize=12, pad=20)
plt.ylabel('Age (years)', fontsize=10)

# 4. Empty subplot for legend/space
plt.subplot(1, 4, 4)
plt.axis('off')  # Hide axes for the empty subplot

# Adjust layout and display
plt.tight_layout(pad=3)
plt.show()

# %% [markdown]
# **Insight:**
# 
# * most of the height lies between 140 and 185
# * most of the weight lies between 40 and 115
# * most of the ages lies between 38 and 68 years
# 

# %% [markdown]
# **FEATURE ENGINEERING**

# %% [markdown]
# 
# Handling skewed data and outliers

# %%
# separate categorical data from numerical data
category = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
cat_data = df[category]
cat_data

# %%
# We create an dataframe to add and test run our new features
numeric = ['age', 'height', 'weight','ap_hi', 'ap_lo']
num_data = df[numeric]
num_data

# %% [markdown]
# **View your numerical data and create normally distributed dataset**

# %%
# create the list of numeric columns
subjective = ['age', 'height', 'weight']

# %%
import matplotlib.pyplot as plt

# Set style for better visuals
#plt.style.use('seaborn')

# Loop through columns and plot
for col in subjective:
    # Create figure with adjusted size
    plt.figure(figsize=(6, 5))
    
    # Plot histogram with customization
    plt.hist(num_data[col], 
             bins=50,
             color='#1f77b4',  # Matplotlib default blue
             edgecolor='white',
             alpha=0.8)
    
    # Add labels and title
    plt.title(f'Distribution of {col}', fontsize=14, pad=20)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add grid and adjust layout
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plt.show()

# %% [markdown]
# **Insight**
# 
# * The height is normally ditributed
# * The weight is slightly skewed to the right
# * The Age is normal
# 
# 

# %% [markdown]
# **Previously, we found out that there are some large vaLues in ap_hi and ap_lo columns and also negative values. Now, we need to handle it.**

# %%
# check columns with negative values of ap_hi (diastole)
num_data[num_data['ap_hi'] < 0]

# %%
# change all the negative values to positive
for values in range(0, len(num_data)):
    if num_data['ap_hi'][values] < 0:
        num_data['ap_hi'][values] = num_data['ap_hi'][values] * -1

# check if we still have negative values
num_data[num_data['ap_hi'] < 0]

# %%
# check columns with negative values of ap_lo (systole)
num_data[num_data['ap_lo'] < 0]

# %%
# change all the negative values to positive
for values in range(0, len(num_data)):
    if num_data['ap_lo'][values] < 0:
        num_data['ap_lo'][values] = num_data['ap_lo'][values] * -1

# check if we still have negative values
num_data[num_data['ap_lo'] < 0]

# %% [markdown]
# **Handle outliers with IQR (interquatile range)**

# %%
# calculate interquartile range
Q1 = num_data.quantile(0.25)
Q3 = num_data.quantile(0.75)
IQR = Q3 -Q1
IQR

# %%
# Get outliers
upper = (num_data > (Q3 + 1.5*IQR))
lower = (num_data < (Q1 - 1.5*IQR))
outliers = upper | lower
outliers.sum()

# %%
# Iterate through Q1, Q3 values and column names simultaneously
for Q1, Q3, column in zip(Q1, Q3, num_data.columns):
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Cap upper outliers: Values > Q3 + 1.5*IQR are set to Q3 + 1.5*IQR
    num_data.loc[(num_data[column] > (Q3 + 1.5*IQR)), column] = Q3 + 1.5*IQR
    
    # Cap lower outliers: Values < Q1 - 1.5*IQR are set to Q1 - 1.5*IQR
    num_data.loc[(num_data[column] < (Q1 - 1.5*IQR)), column] = Q1 - 1.5*IQR

# Return the modified DataFrame with capped outliers
num_data

# %%
bp = ['ap_hi', 'ap_lo']
for columns in bp:
    plt.hist(num_data[columns], bins=30)
    plt.xlabel(columns)
    plt.show()

# %%
import matplotlib.pyplot as plt

# Set style for better visuals
#plt.style.use('seaborn')

# Loop through columns
for col in subjective:
    # Create figure
    plt.figure(figsize=(7, 5))
    
    # Create histogram with customizations
    plt.hist(num_data[col], 
             bins=30,
             color='#1f77b4',  # Nice blue color
             edgecolor='white', # White edges on bars
             alpha=0.8)        # Slight transparency
    
    # Add labels and title
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add grid lines
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# %%
# Add new features using vectorized operations (more efficient than loops)
num_data['bmi'] = num_data['weight'] / (num_data['height']/100)**2
num_data['age_year'] = num_data['age'] / 365

# Hypertensive stage classification
conditions = [
    (num_data['ap_hi'] > 180) | (num_data['ap_lo'] > 120),
    (num_data['ap_hi'] > 139) | (num_data['ap_lo'] > 90),
    (num_data['ap_hi'] > 129) | (num_data['ap_lo'] > 80),
    (num_data['ap_hi'] > 119),
    True  # Default case
]
choices = [4, 3, 2, 1, 0]
num_data['hypertensive'] = np.select(conditions, choices)

# Age group classification
age_bins = [0, 15, 20, 40, 60, float('inf')]
age_labels = ['children', 'teenager', 'youth', 'middle_age', 'elderly']
num_data['age_group'] = pd.cut(num_data['age_year'], bins=age_bins, labels=age_labels)

# Obesity classification
bmi_bins = [0, 25, 30, 35, 40, float('inf')]
num_data['obesity'] = pd.cut(num_data['bmi'], bins=bmi_bins, labels=[0, 1, 2, 3, 4])

num_data

# %%
num_data['age_group'].unique()

# %%
# encode the category in num_data
num_data = pd.get_dummies(num_data, dtype = int)
num_data

# %%
print(num_data.columns.tolist())


# %%
print(cat_data.columns.tolist())


# %%
# Reload the original CSV
original_data = pd.read_csv('cardio_train.csv', sep=';')

# Verify gender column exists
if 'gender' in original_data.columns:
    cat_data['gender'] = original_data['gender']  # Add it back
    print("Gender column restored!")
else:
    print("Gender not found in original data - check column names")

# %%
print(cat_data.columns.tolist())

# %%
print(cat_data.columns.tolist())

# %%
print(cat_data.head(5))

# %%
# 1. First check if gender column exists
if 'gender' in cat_data.columns:
    # 2. Convert numeric to labels (only if values are 1/2)
    cat_data['gender'] = cat_data['gender'].map({1: 'Female', 2: 'Male'})
    

    
else:
    print("Warning: 'gender' column not found - skipping processing")

# Show result
print(cat_data.head())

# %%
cat_data = pd.get_dummies(cat_data, dtype = int)

cat_data


# %% [markdown]
# **Joining the both the categorical data and numeric data into a single table**

# %%
#Visualizing old data table
df

# %%
# new data
data = pd.concat([num_data,cat_data], axis =1)
data

# %% [markdown]
# **Preparation of Data for Model Training**

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# %%
features = data.drop('cardio', axis = 1)
target = data['cardio']

# %%
features

# %%
target

# %%
print(features.columns.tolist())

# %%
parameter = features.drop(
    ['age_group_children', 'age_group_teenager', 'age_group_youth', 'age_group_middle_age', 
     'age_group_elderly', 'obesity_0', 'obesity_1', 'obesity_2', 'obesity_3', 
     'obesity_4', 'hypertensive'], 
    axis=1
)

parameter


# %%
feature_names = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'age_year',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender_Female',
       'gender_Male']
print(feature_names)


# %%
#features_scale = scaler.fit_transform(parameter)

# %%
#x_scale = pd.DataFrame(features_scale,columns=[parameter.columns])
x_scale

# %%
from sklearn.model_selection import train_test_split
# Split data
X_train,X_test,y_train,y_test = train_test_split(parameter,target,test_size= 0.3, shuffle = True, random_state = 42,  stratify=target)



# %%


# %% [markdown]
# **Normalization using StandardScaler** 
# 
# This is to prevent features with large ranges from dominating the learning process

# %%
from sklearn.preprocessing import StandardScaler

#Standardizing the feature

##scaler = StandardScaler()
#X_train_scaled = scaler.fit(pd.DataFrame(X_train, columns=feature_names))

#X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# **Feature Selection Using ANOVA**
# 
# This will help select most relevant feature for training the model

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# 1) Suppose X_train, X_test are your raw feature arrays (n_samples × n_features)
#    and y_train is your label array (n_samples,)

# 2) Wrap them in DataFrames if you like (to keep track of names)
feature_names = [
    'age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'age_year',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'gender_Female', 'gender_Male'
]
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df  = pd.DataFrame(X_test,  columns=feature_names)

# 3) Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)  # <-- this returns an array
X_test_scaled  = scaler.transform(X_test_df)

# sanity check shapes
print(X_train_scaled.shape)  # e.g. (56000, 14)

# 4) Select top k features by ANOVA F-test
k = 3
anova_selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = anova_selector.fit_transform(X_train_scaled, y_train)
X_test_selected  = anova_selector.transform(X_test_scaled)

# 5) Inspect scores & names
anova_scores = anova_selector.scores_
anova_results = dict(zip(feature_names, anova_scores))

print("ANOVA F-scores:")
for name, score in anova_results.items():
    print(f" • {name:15s}: {score:.2f}")

# 6) Which features were picked?
selected_idx = anova_selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_idx]
print("\nTop-3 features:", selected_features)


# %%
#!pip install catboost

# %%
#!pip install xgboost

# %%
from sklearn.ensemble import  GradientBoostingClassifier 



gbc = GradientBoostingClassifier(random_state=14)
gbc.fit(X_train_scaled, y_train)




# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics as sm
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Dictionary of trained models
models = {
    'Gradient Boosting': gbc,
    
}

# Dictionary to store evaluation metrics
evaluation_results = {}

# Collect metrics for each model
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Evaluating {name}")
    print(f"{'='*50}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    conf_matrix = sm.confusion_matrix(y_test, y_pred)
    accuracy = sm.accuracy_score(y_test, y_pred)
    precision = sm.precision_score(y_test, y_pred)
    recall = sm.recall_score(y_test, y_pred)
    f1 = sm.f1_score(y_test, y_pred)
    roc_auc = sm.roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    # Store results
    evaluation_results[name] = {
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    # Visualization - Confusion Matrix (individual for each model)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Create comparison DataFrame
metrics_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
metrics_df = metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
print("\nModel Performance Comparison:")
print(metrics_df.sort_values('roc_auc', ascending=False))

# Create stacked comparison plots for each metric
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
model_names = list(models.keys())

# Set up the figure and axes
fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=(10, 6*len(metrics)))

# If only one metric, axes won't be an array
if len(metrics) == 1:
    axes = [axes]

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Get values for both models
    values = [evaluation_results[model][metric] for model in model_names]
    
    # Create bar plot
    bars = ax.bar(model_names, values, color=['skyblue', 'salmon'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)  # Assuming all metrics are between 0 and 1
    
    # Add horizontal line at 1 for reference
    ax.axhline(1, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# Optional: Plot all metrics in one figure for quick comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

for i, model in enumerate(model_names):
    values = [evaluation_results[model][metric] for metric in metrics]
    plt.bar(x + (i * width), values, width, label=model)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x + width/2, [m.capitalize() for m in metrics])
plt.legend()
plt.ylim(0, 1.1)

# Add value labels
for i, model in enumerate(model_names):
    for j, metric in enumerate(metrics):
        value = evaluation_results[model][metric]
        plt.text(j + (i * width), value + 0.02, f'{value:.3f}', ha='center')

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score,
                           confusion_matrix, classification_report)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate model performance on both training and test sets
    to check for overfitting.
    """
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Get probabilities if available (for ROC AUC)
    train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
    test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics for training set
    train_metrics = {
        'accuracy': accuracy_score(y_train, train_pred),
        'precision': precision_score(y_train, train_pred),
        'recall': recall_score(y_train, train_pred),
        'f1': f1_score(y_train, train_pred),
        'roc_auc': roc_auc_score(y_train, train_proba) if train_proba is not None else None,
        'confusion_matrix': confusion_matrix(y_train, train_pred)
    }
    
    # Calculate metrics for test set
    test_metrics = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred),
        'recall': recall_score(y_test, test_pred),
        'f1': f1_score(y_test, test_pred),
        'roc_auc': roc_auc_score(y_test, test_proba) if test_proba is not None else None,
        'confusion_matrix': confusion_matrix(y_test, test_pred)
    }
    
    return {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfit_score': {
            metric: train_metrics[metric] - test_metrics[metric] 
            for metric in ['accuracy', 'precision', 'recall', 'f1']
        }
    }

def plot_overfitting_diagnosis(results):
    """
    Visualize the performance difference between train and test sets
    to diagnose overfitting.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = [res['model_name'] for res in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get train and test scores for all models
        train_scores = [res['train_metrics'][metric] for res in results]
        test_scores = [res['test_metrics'][metric] for res in results]
        
        # Bar width
        bar_width = 0.35
        
        # X-axis positions
        x = range(len(model_names))
        
        # Plot bars
        train_bars = ax.bar([pos - bar_width/2 for pos in x], train_scores, 
                           bar_width, label='Train', color='skyblue')
        test_bars = ax.bar([pos + bar_width/2 for pos in x], test_scores, 
                          bar_width, label='Test', color='salmon')
        
        # Add value labels
        for bars in [train_bars, test_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Score')
        ax.legend()
        ax.set_ylim(0, 1.1)
    
    plt.suptitle('Train vs Test Performance - Overfitting Diagnosis', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Plot overfit scores (difference between train and test)
    plt.figure(figsize=(10, 6))
    
    overfit_data = []
    for res in results:
        for metric, diff in res['overfit_score'].items():
            overfit_data.append({
                'Model': res['model_name'],
                'Metric': metric.capitalize(),
                'Train-Test Difference': diff
            })
    
    overfit_df = pd.DataFrame(overfit_data)
    
    sns.barplot(data=overfit_df, x='Metric', y='Train-Test Difference', 
                hue='Model', palette='viridis')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Overfitting Indicators (Train Performance - Test Performance)')
    plt.ylabel('Performance Difference')
    plt.legend(title='Model')
    plt.show()

# Dictionary of trained models
models = {
    'Gradient Boosting': gbc,
    
}

# Evaluate all models
results = []
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Evaluating {name}")
    print(f"{'='*50}")
    
    result = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, name)
    results.append(result)
    
    # Print detailed metrics
    print("\nTraining Set Performance:")
    print(f"Accuracy: {result['train_metrics']['accuracy']:.4f}")
    print(f"Precision: {result['train_metrics']['precision']:.4f}")
    print(f"Recall: {result['train_metrics']['recall']:.4f}")
    print(f"F1-Score: {result['train_metrics']['f1']:.4f}")
    if result['train_metrics']['roc_auc'] is not None:
        print(f"ROC AUC: {result['train_metrics']['roc_auc']:.4f}")
    
    print("\nTest Set Performance:")
    print(f"Accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Precision: {result['test_metrics']['precision']:.4f}")
    print(f"Recall: {result['test_metrics']['recall']:.4f}")
    print(f"F1-Score: {result['test_metrics']['f1']:.4f}")
    if result['test_metrics']['roc_auc'] is not None:
        print(f"ROC AUC: {result['test_metrics']['roc_auc']:.4f}")
    
    print("\nOverfitting Indicators (Train - Test):")
    for metric, diff in result['overfit_score'].items():
        print(f"{metric.capitalize()}: {diff:.4f}")
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(result['train_metrics']['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=ax1)
    ax1.set_title(f'Train Confusion Matrix - {name}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Truth')
    
    sns.heatmap(result['test_metrics']['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=ax2)
    ax2.set_title(f'Test Confusion Matrix - {name}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Truth')
    
    plt.tight_layout()
    plt.show()

# Generate comparison visualizations
plot_overfitting_diagnosis(results)

# Create a summary dataframe
summary_data = []
for res in results:
    for set_type in ['train_metrics', 'test_metrics']:
        row = {'Model': res['model_name'], 'Dataset': set_type.replace('_metrics', '')}
        for metric, value in res[set_type].items():
            if metric != 'confusion_matrix':
                row[metric] = value
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\nPerformance Summary:")
print(summary_df)

# %%
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime
import pandas as pd

# Assuming you have these variables from your training code:
# X_train_scaled, y_train, gbc

# 1. Save the Gradient Boosting model with metadata
model_info = {
    'model': gbc,
    'model_name': 'GradientBoostingClassifier',
    'model_params': gbc.get_params(),
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'features': list(X_train_scaled.columns) if hasattr(X_train_scaled, 'columns') else f'feature_0_to_{X_train_scaled.shape[1]-1}',
    'target': 'your_target_variable_name',  # Replace with your actual target name
    'input_shape': X_train_scaled.shape,
    'classes': gbc.classes_,
    'n_classes': gbc.n_classes_,
    'training_score': gbc.score(X_train_scaled, y_train)
}

# Save the model with metadata
dump(model_info, 'gradient_boosting_model_with_metadata_3.joblib')

# 2. Save just the model (minimal version)
dump(gbc, 'gradient_boosting_model_3.joblib')

# Save the scaler
dump(scaler, 'scaler_2.joblib')


print("Models saved successfully:")
print("- gradient_boosting_model_3.joblib")
print("- gradient_boosting_model_with_metadata_3.joblib")
print("- StandardScaler.joblib")

# 4. Verification code (optional)
def verify_model(filepath):
    from joblib import load
    data = load(filepath)
    print(f"\nVerifying {filepath}:")
    print(f"Model: {data.get('model_name', 'Unknown')}")
    print(f"Training date: {data.get('training_date', 'Unknown')}")
    print(f"Features: {data.get('features', 'Unknown')}")
    print(f"Target: {data.get('target', 'Unknown')}")
    print(f"Training score: {data.get('training_score', 'Unknown')}")
    if 'model' in data:
        print("Model object loaded successfully")

# Verify the saved models
verify_model('gradient_boosting_model_with_metadata_2.joblib')



# %%



