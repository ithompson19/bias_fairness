from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression

from data_reader import Adult


def main(args):
    # introduce bias in labels
    # get confidence of decision, and flip low confidence rows?
    
    training_data, training_labels, training_sensitive_attributes = Adult.training_data()
    print(training_data.head())
    test_data, test_labels, test_sensitive_attributes = Adult.test_data()
    print(test_data.head())
    
    estimator = LogisticRegression(max_iter = 10000, n_jobs = -1)
    
    estimator.fit(X = training_data, y = training_labels)
    prediction = estimator.predict(X = test_data)
    
    constraint = DemographicParity(difference_bound = 0.01)
    
    mitigator = ExponentiatedGradient(estimator = estimator, constraints = constraint)
    
    mitigator.fit(X = training_data, y = training_labels, sensitive_features = training_sensitive_attributes)
    
    prediction_mitigated = mitigator.predict(X = test_data)
    
    print(prediction)
    
    print(prediction_mitigated)
main(None)