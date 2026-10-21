"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.array(X)




   
    means = np.nanmean(X,axis=0)
    means = np.where(np.isnan(means),0,means)
    X = np.where(np.isnan(X),means,X)

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q_1 = np.percentile(X,25,axis=0)
    q_3 = np.percentile(X,75,axis=0)
    IQR = q_3 - q_1

    lower = q_1 - k * IQR
    upper = q_3 + k * IQR

    return(lower,upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X,lower,upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator / (denominator + eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack((X,col))

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    unique = np.unique(labels)
    return (labels[:, None] == unique).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    std = np.where(std != 0,std,1)

    return(mean,std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    return np.hstack([np.ones((X.shape[0],1)),X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    np.random.seed(seed)
    return np.random.permutation(np.arange(n_samples))

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return (indices[0:n_train],indices[n_train:n_train+n_val],indices[n_train+n_val:])

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return(X[indices],y[indices])

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T @ X
    b = X.T @ y

    theta = np.linalg.solve(A,b)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return float(np.mean(np.abs(y_pred - y_true)))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    
    ss_res = np.sum((y_true - y_pred)**2) 
    ss_tot = np.sum((y_true - y_true.mean())**2) 

    if ss_tot == 0.0:
        return ss_tot
    else:
        return 1 - (ss_res / ss_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred

    return {
        "mean" : float(np.mean(r)),
        "std"  : float( np.std(r)),
        "median_abs" : float(np.median(np.abs(r)))
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = impute_nan_with_mean(X)
    lower,upper = compute_iqr_bounds(X,iqr_k)
    X = clip_columns(X,lower,upper)
    return X

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    numerator = X_num[:,ratio_num_idx]
    denominator = X_num[:,ratio_den_idx]
    ratio = make_ratio_feature(numerator,denominator)
    fm = append_column(X_num,ratio)
    if cat_labels is not None:
        one_hot = one_hot_encode(cat_labels)
        fm = np.hstack([fm,one_hot])
        
    return fm

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_samples = X.shape[0]
    indices = make_shuffled_indices(n_samples, seed)
    train_i,val_i,test_i = partition_indices(indices,train_ratio,val_ratio)
    X_train,y_train = subset_xy(X,y,train_i)
    X_val,y_val = subset_xy(X,y,val_i)
    X_test,y_test = subset_xy(X,y,test_i)

    return {
        "X_train":X_train,
        "y_train":y_train,
        "X_val":X_val,
        "y_val":y_val,
        "X_test":X_test,
        "y_test" :y_test,
        
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    mean, std = fit_standardizer(splits['X_train'])

    std_splits = splits.copy()

    for name  in [ 'X_train','X_val','X_test']:
        X =  apply_standardizer(splits[name], mean, std)
        std_splits[name] = add_bias_column(X)
    
    return(std_splits,mean,std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    return {
        'mae':mean_absolute_error(y_true,y_pred),
        'rmse':root_mean_squared_error(y_true,y_pred),
        'r2':r_squared(y_true,y_pred),
        'residual_summary':residual_summary(y_true,y_pred)
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)
    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels)
    splits = make_train_val_test(X_feat, y, train_ratio, val_ratio, seed) 
    std_splits, _, _ = standardize_and_add_bias(splits) 
    theta, *_ = np.linalg.lstsq(std_splits['X_train'], std_splits['y_train'], rcond=None) 
    # The previous implemented method passed all test but here rises a singularity error.
    y_val_pred = ols_predict(std_splits['X_val'],theta)
    y_test_pred = ols_predict(std_splits['X_test'],theta)
    val_metrics = evaluate_predictions(std_splits['y_val'], y_val_pred)
    test_metrics = evaluate_predictions(std_splits['y_test'], y_test_pred)

    return{
        "test_metrics" : test_metrics,
        "theta" : theta,
        "val_metrics" : val_metrics,
        "y_test" :std_splits['y_test'],
        "y_test_pred" : y_test_pred
    }

