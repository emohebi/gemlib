from sklearn.model_selection import cross_val_score


def get_cross_validation_average_score(clf, X, y, cross_val=5):
    scores = cross_val_score(clf, X, y, cv=cross_val)
    return scores.mean()
