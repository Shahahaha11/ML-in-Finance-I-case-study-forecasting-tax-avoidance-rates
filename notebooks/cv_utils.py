from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import pandas as pd

def grid_CV(x, y, model, param_grid, display_res=False):

    # rolling CV splitter (same logic as before)
    splits = []
    train_idx = [0, 1452]
    valid_idx = [1452, 1815]

    for _ in range(6):
        splits.append((
            np.arange(train_idx[0], train_idx[1]),
            np.arange(valid_idx[0], valid_idx[1])
        ))
        train_idx = [0, valid_idx[1]]
        valid_idx = [train_idx[1], valid_idx[1] + 363]

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    gs = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=splits,
        scoring=scorer,
        n_jobs=-1,
        refit=True
    )

    gs.fit(x.values, y.values.ravel())

    # compute fold-wise RMSE for best estimator
    train_score, valid_score = [], []
    best_est = gs.best_estimator_

    for tr, va in splits:
        Xtr, ytr = x.iloc[tr], y.iloc[tr]
        Xva, yva = x.iloc[va], y.iloc[va]

        best_est.fit(Xtr.values, ytr.values.ravel())
        train_score.append(np.sqrt(mean_squared_error(ytr, best_est.predict(Xtr.values))))
        valid_score.append(np.sqrt(mean_squared_error(yva, best_est.predict(Xva.values))))

    if display_res:
        view = pd.DataFrame({"cv_train": train_score, "cv_val": valid_score})
        return train_score, valid_score, view, gs.best_params_
    else:
        return train_score, valid_score, gs.best_params_
