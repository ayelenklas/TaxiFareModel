from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import haversine_distance, haversine_vectorized, compute_rmse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib


class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude']), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        pipeline= Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])
        self.pipeline = pipeline

    def run(self):
        """set and train the pipeline"""

        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        MLFLOW_URI = "https://mlflow.lewagon.co/"
        EXPERIMENT_NAME = "[DE] [Berlin] [ayelenklas] TaxiFareModel v1"
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


    def mlflow_log_param(self, param_name, param_value):
        self.mlflow_log_param(param_name, param_value)


    def mlflow_log_metric(self, metric_name, metric_value):
        self.mlflow_log_metric(metric_name, metric_value)

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'TaxiFareModel.joblib')


if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.loc[:,"pickup_datetime":"passenger_count"]
    y = df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    print(trainer.evaluate(X_test,y_test))
    trainer.save_model()
    print("---------END---------")

    # experiment_id = trainer.mlflow_experiment_id
    # print(
    #     f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}"
    # )
