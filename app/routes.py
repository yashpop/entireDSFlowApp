from app import app, db
from app.models import Experiment
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.model_selection import train_test_split
import os.path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from flask_restful import abort

#storing and retrieving models
import pickle
from sklearn.externals import joblib

global iris_class_data_ids
iris_class_data_ids = set([1,2,3,4])

@app.route("/", methods=["POST"])
def create():
    data = request.get_json()
    start_date = datetime.now()
    experiment = Experiment(name_=data.get("name"), start_date_=start_date, type_=data.get("type"), test_data_=data.get("test_data"), train_data_=data.get("train_data"))
    db.session.add(experiment)
    db.session.flush()
    db.session.commit()
    return jsonify(experiment_to_dict(experiment)), 201


@app.route("/", methods=["GET"], defaults={'experiment_id': None})
@app.route("/<experiment_id>", methods=["GET"])
def retrieve(experiment_id):
    if experiment_id:
        if Experiment.query.filter_by(id_=experiment_id).count() > 0:
            experiment = Experiment.query.filter_by(id_=experiment_id).first()
            return jsonify(experiment_to_dict(experiment))
        else:
            abort(404, message="Specified resource does not exist.")
    else:
        experiments = Experiment.query.all()
        return jsonify([experiment_to_dict(experiment) for experiment in experiments])


@app.route("/<experiment_id>", methods=["PUT"])
def update(experiment_id):
    if Experiment.query.filter_by(id_=experiment_id).count() > 0:
        experiment = Experiment.query.filter_by(id_=experiment_id).first()
        data = request.get_json()
        if data.get("name"):
            experiment.name_ = data.get("name")
        if data.get("type"):
            experiment.type_ = data.get("type")
        if data.get("training_data") or data.get("testing_data"):
            experiment.result_ = ""
            if data.get("training_data"):
                experiment.training_data_ = data.get("training_data")
            if data.get("testing_data"):
                experiment.testing_data_ = data.get("testing_data")
    else:
        abort(404, message="Specified resource does not exist.")
    db.session.add(experiment)
    db.session.flush()
    db.session.commit()
    return jsonify(experiment_to_dict(experiment)), 201


@app.route("/<experiment_id>", methods=["DELETE"])
def delete(experiment_id):
    if Experiment.query.filter_by(id_=experiment_id).count() > 0:
        Experiment.query.filter_by(id_=experiment_id).delete()
        db.session.commit()
        return "", 204
    else:
        abort(404, message="Specified resource does not exist.")



@app.route("/train/<experiment_id>")
def train(experiment_id):
    try:
        experiment = retrieve(experiment_id)
        train_data = experiment['train_data']
        type = experiment['type']
        X = train_data[:,0:4].astype(float)
        Y = train_data[:,4:]
        if type in "Neural Networks":
            # convert integers to dummy variables (i.e. one hot encoded)
            y = np_utils.to_categorical(y)
            model = neural_net_model()
        else:
            model = KNeighborsClassifier(n_neighbors=8)
            model.fit(X, y)
        joblib.dump(model,str(experiment_id)+'.sav')
    except error as e:
        abort(404, message="Specified resource does not exist.")

    return jsonify(Success="Successfully trained and saved the specified model" )


@app.route("/test/<experiment_id>",methods=['GET'])
def test(experiment_id):
    try:
        experiment_id = int(experiment_id)
        experiment = retrieve(experiment_id)
        test_data = experiment['test_data']
        model = joblib.load(str(experiment_id)+'.sav')
        X_test = test_data[:,0:4].astype(float)
        y_test = test_data[:,4:]
        results = predict_with_trained_model(model,X_test,y_test)

        return jsonify({"accuracy_score":results[0], "predicted_data":results[1]} )
    except ValueError:
        return jsonify(Error="Please make sure of user ID is number")
    except error as e:
        abort(404, message="Specified resource does not exist.")


@app.route("/predict/<experiment_id>",methods=['GET'])
def predict(experiment_id):
    try:
        experiment_id = int(experiment_id)
    except ValueError:
        return jsonify(Error="Please make sure of user ID is number")
    model = joblib.load(str(experiment_id)+'.sav')
    experiment = retrieve(experiment_id)
    test_data = experiment['test_data']
    results = predict_with_trained_model(model,X_test,y_test)
    return jsonify({"accuracy_score":results[0],"predicted_data":results[1]})

def experiment_to_dict(experiment):
    return {
        "id": experiment.id_,
        "name": experiment.name_,
        "start_date": experiment.start_date_,
        "type": experiment.type_,
        "result": experiment.result_,
        "test_data": experiment.test_data_,
        "train_data": experiment.train_data_
    }


def get_train_data(experiment_id):

    if experiment_id in iris_class_data_ids:
        if os.path.isfile('iris_train_data.csv'):
            dataframe = pd.read_csv('iris_data.csv',header=None)
            #dataset = dataframe.values
            #X = dataset[:,0:4].astype(float)
            #Y = dataset[:,4:]
            return dataset
        else:
            X,y = get_iris_class_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
            df = pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)],axis=1)
            df.to_csv('iris_train_data.csv',header=False,index=False)
            return df
    else:
        return None

def get_test_data(experiment_id):

    if experiment_id in iris_class_data_ids:
        if os.path.isfile('iris_test_data.csv'):
            dataframe = pd.read_csv('iris_test_data.csv',header=None)
            dataset = dataframe.values
            #X = dataset[:,0:4].astype(float)
            #Y = dataset[:,4:]
            return dataset
        else:
            dataset = get_iris_class_data()
            X = dataset[:,0:4]
            Y = dataset[:,4]
            X,y = wrangle_data(X,Y)
            pd.DataFrame()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
            df = pd.concat([pd.DataFrame(X_test),pd.DataFrame(y_test)],axis=1)
            df.to_csv('iris_test_data.csv',header=False,index=False)
            return df

def get_iris_class_data():
    dataframe = pd.read_csv('iris_data.csv',header=None)
    dataset = dataframe.values
    return dataset

def wrangle_data(X,Y):
    X = dataset[:,0:4].astype(float)
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    return X,encoded_y

# define neural network model model
def neural_net_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def predict_class_with_trained_model(model,X_test,y_test):
    y_predict = model.predict(X_test)
    return (accuracy_score(y_test,y_pred),y_predict)
