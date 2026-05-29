import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class5\assignment1\Decision_Tree.csv")
label_encoder_temperature=LabelEncoder()
label_encoder_vibration=LabelEncoder()
label_encoder_failure=LabelEncoder()
data["Temperature"]=label_encoder_temperature.fit_transform(data["Temperature"])
data["Vibration"]=label_encoder_vibration.fit_transform(data["Vibration"])
data["Failure"]=label_encoder_failure.fit_transform(data["Failure"])

x=data[["Temperature", "Vibration"]]
y=data["Failure"]

model=DecisionTreeClassifier(criterion="entropy")

model.fit(x,y)

temp = label_encoder_temperature.transform(['High'])[0]
vib = label_encoder_vibration.transform(['Medium'])[0]

prediction = model.predict([[temp, vib]])

if prediction[0] == 1:
    print("Machine will Fail")
else:
    print("Machine will not Fail")
    

plt.figure(figsize=(8,10))
tree.plot_tree(
    model,
    feature_names=["Temperature", "Vibration"],
    class_names=["No", "Yes"]
)
plt.title("decision tree")
plt.show()