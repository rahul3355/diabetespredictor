# s1 : import the lib
import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect, url_for
from sqlite3 import *
import os

currentdirectory = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/check", methods=["POST", "GET"])
def check():
    # s2 : load the data
    data = pd.read_csv("diabetes.csv")
    print(data.head())

    # s3 : understanding the data
    data[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data[[
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

    data["Glucose"].fillna(data["Glucose"].mean(), inplace=True)
    data["BloodPressure"].fillna(data["BloodPressure"].mean(), inplace=True)
    data["SkinThickness"].fillna(data["SkinThickness"].mean(), inplace=True)
    data["Insulin"].fillna(data["Insulin"].mean(), inplace=True)
    data["BMI"].fillna(data["BMI"].mean(), inplace=True)
    print(data.isnull().sum())

    # s4 : features and target
    features = data.drop("Outcome", axis="columns")
    target = data['Outcome']
    print(features.head())
    print(target.head())

    scaler = MinMaxScaler()
    new_features = scaler.fit_transform(features)

    # s5 : find the value of k
    x_train, x_test, y_train, y_test = train_test_split(new_features, target)
    model = KNeighborsClassifier(n_neighbors=17, metric='euclidean')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    cr = classification_report(y_test, y_pred)
    print(cr)

    pid = int(request.args.get("pid"))
    name = request.args.get("name")

    age = int(request.args.get("age"))
    glucose = float(request.args.get("glucose"))
    preg = int(request.args.get("preg"))
    bp = float(request.args.get("bp"))
    st = float(request.args.get("st"))
    insulin = float(request.args.get("insulin"))
    bmi = float(request.args.get("bmi"))
    dpf = float(request.args.get("dpf"))

    # backend
    # if request.method=="POST":
    #     con=None
    #     try:
    #         con = connect("diabetes1.db")
    #         cursor = con.cursor()
    #         sql = "insert into patient values('%d','%s','%d','%f','%f','%f','%f','%f','%f','%d')"
    #         cursor.execute(sql%(pid,name,preg,glucose,bp,st,insulin,bmi,dpf,age))
    #         con.commit()
    #         return render_template("home.html",msg="record inserted")
    #     except Exception as e:
    #         con.rollback()
    #         return render_template("home.html",data="issue"+str(e))
    #     finally:
    #         if con is None:
    #             con.close()

    # prediction calculation
    dd = []
    dd.append(preg)
    dd.append(glucose)
    dd.append(bp)
    dd.append(st)
    dd.append(insulin)
    dd.append(bmi)
    dd.append(dpf)
    dd.append(age)
    print(dd)
    ddd = [dd]

    # fetching precision
    precision, recall, fscore, support = score(y_test, y_pred, average='macro')
    print('Precision : {}'.format(precision))
    print(precision)
    pre = round(precision, 2)
    print(pre)
    prec = pre * 100

    # prediction
    t_d = scaler.transform(ddd)
    pdd = model.predict(t_d)
    outcome = int(pdd[0])
    print(pdd)
    pddd = ""

    connection = sqlite3.connect(currentdirectory + "\diabetes1.db")
    cursor = connection.cursor()
    query1 = "INSERT INTO patient VALUES('{pid}','{name}','{preg}','{glucose}','{bp}','{st}','{insulin}','{bmi}','{dpf}','{age}','{outcome}')".format(
        pid=pid, name=name, preg=preg, glucose=glucose, bp=bp, st=st, insulin=insulin, bmi=bmi, dpf=dpf, age=age, outcome=outcome)
    cursor.execute(query1)
    connection.commit()
    print("Entry inserted")
    connection.close()

    if pdd[0] == 1:
        pddd = "The given patient has diabetes.  " + str(prec) + "% precise"
    else:
        pddd = "The given patient does not have diabetes.  " + \
            str(prec) + "% precise"
    print(pddd)
    msg = " " + pddd + ". record saved."
    return render_template("home.html", msg=msg)


@app.route("/table", methods=["GET"])
def table():
    try:    
        if request.method == "GET":
            connection = sqlite3.connect(currentdirectory + "\diabetes1.db")
            cursor = connection.cursor()
            query1 = "SELECT * FROM patient"
            result = cursor.execute(query1)
            result = result.fetchall()
            return render_template("table.html", data=result)
    except:
        return render_template("table.html", data=" $$ ")
    finally:
        if connection is None:
                connection.close()
            
@app.route("/delp/<int:id>")
def delp(id):
	con = None
	try:
		con = connect("diabetes1.db")
		cursor = con.cursor()
		sql = "delete from patient where pid='%d'"
		cursor.execute(sql%(id))
		con.commit()
	except Exception as e:
		con.rollback()
		return render_template("table.html", msg="issue"+str(e))
	finally:
		if con is not None:
			con.close()
		return redirect(url_for('table'))


@app.route("/working")
def working():
    return render_template("working.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


if __name__ == "__main__":
    app.run(debug=True)
