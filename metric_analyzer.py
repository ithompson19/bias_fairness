
from typing import List, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate
import constants as const
from data_reader import DataReader
import file_handler

def generate_metrics(data_reader: DataReader,
                     tests: List[Tuple[Tuple[str, str, float, float], float]]):
    try:
        file_handler.read_metrics(tests)
        print('Metrics found')
        return
    except:
        pass
    print('\nAnalyzing model metrics')
    test_data, test_labels, test_sensitive_attributes = data_reader.test_data()
    all_models = file_handler.read_models(tests)
    results: pd.DataFrame = pd.DataFrame()
    for trial_num, model_test_groups in all_models:
        for model_test_group, (flip_rate, confidence_threshold) in zip(model_test_groups, tests):
            predictions = [model.predict(X = test_data) for model in model_test_group]
            
            model_metrics = [accuracy_score(y_true = test_labels, y_pred = prediction) for prediction in predictions]
            
            for metric_function, prediction in zip([metric_function for (_, (metric_function, _)) in const.CONSTRAINED_MODELS.items()], predictions[1:]):
                if metric_function == true_positive_rate or metric_function == false_positive_rate:
                    metric_frame = MetricFrame(metric_function,
                                                y_true = test_labels,
                                                y_pred = prediction,
                                                sensitive_features = test_sensitive_attributes)
                    model_metrics.append(metric_frame.difference())
                else:
                    model_metrics.append(metric_function(y_true = test_labels,
                                                            y_pred = prediction,
                                                            sensitive_features = test_sensitive_attributes))

            results = results.append(file_handler.generate_metrics_row(data_reader=data_reader,
                                                             trial_num=trial_num,
                                                             flip_rate=flip_rate,
                                                             confidence_threshold=confidence_threshold,
                                                             model_metrics=model_metrics), ignore_index=True)
    file_handler.save_metrics(tests, results)