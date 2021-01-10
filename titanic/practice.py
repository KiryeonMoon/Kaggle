# %%
# 데이터 분석 패키지
# 데이터 씹뜯맛즐하는 패키지
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas
import numpy

# 데이터를 시각화하는 도구같은 패키지
import seaborn
import matplotlib.pyplot

# 머신러닝 툴 패키지
# 툴은 알고리즘 따라 여러가지가 있으니 전부 소환
# 옛날부터 머신러닝이 발전해오면서 연구자들이 오픈 커뮤니티에 축적시켜온 자료들
# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 데이터 불러와서 변수에 집어넣기
# 데이터를 패널 데이터 도구(Pandas)로 불러와 각각 train, test라는 이름의 변수에 집어넣는다.
train = pandas.read_csv("C:/Users/Be Irreplaceable/Desktop/titanic/train.csv")
test = pandas.read_csv("C:/Users/Be Irreplaceable/Desktop/titanic/test.csv")


def bar_chart(feature):
    survived = train[train["Survived"] == 1][feature].value_counts()
    dead = train[train["Survived"] == 0][feature].value_counts()
    df = pandas.DataFrame([survived, dead])
    df.index = ["Survived", "Dead"]
    df.plot(kind="bar", stacked=True, figsize=(10, 5))


# %%
train = train.drop(["Ticket", "Cabin"], axis=1)
test = test.drop(["Ticket", "Cabin"], axis=1)

combine = [train, test]

# %%
# 이름 외 title이라는 새로운 column 생성
# dataset에 있는 Name 항목의 String을 Extract해라.
for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract("([A-Za-z]+\.)", expand=False)

# %%
for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(
        ["Lady.", "Countess.", "Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer."], "Rare")
    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss.")
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss.")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs.")
# 이름을 다 바꾸었으니, 이제 이름은 손가락으로 셀 수 있는 범주형 데이터로 볼 수 있다.

# %%
# 문자를 숫자로 바꾸는 맵핑 작업 수행
title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

# %%
combine = [train, test]
# %%
mapping_sex = {"male": 0, "female": 1}
for i in combine:
    i["Sex"] = i["Sex"].map(mapping_sex)

# %%
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"))
# %%
train["Age"] = train["Age"].fillna(
    train.groupby("Title")["Age"].transform("median"))
# %%
test["Age"] = test["Age"].fillna(
    test.groupby("Title")["Age"].transform("median"))
# %%
# child : 0
# young : 1
# adult : 2
# mid-age : 3
# senior : 4
for i in combine:
    i.loc[i["Age"] <= 16, "Age"] = 0
    i.loc[(i["Age"] > 16) & (i["Age"] <= 26), "Age"] = 1
    i.loc[(i["Age"] > 26) & (i["Age"] <= 36), "Age"] = 2
    i.loc[(i["Age"] > 36) & (i["Age"] <= 62), "Age"] = 3
    i.loc[i["Age"] > 62, "Age"] = 4

# %%
mapping_embarked = {"S": 0, "C": 1, "Q": 2}
for i in combine:
    i["Embarked"] = i["Embarked"].map(mapping_embarked)

# %%
for i in combine:
    i["Embarked"] = i["Embarked"].fillna(0)

# %%
combine = [train, test]
for dataset in combine:
    dataset.loc[dataset["Fare"] <= 17, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 17) & (dataset["Fare"] <= 30), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 30) & (dataset["Fare"] <= 100), "Fare"] = 2
    dataset.loc[dataset["Fare"] > 100, "Fare"] = 3

test.Fare = test.Fare.fillna(2)
test.info()

# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

# clf = KNeighborsClassifier(n_neighbors=13)
# score = cross_val_score(clf, train, train["Survived"], cv=k_fold,n_jobs=1, scoring="accuracy")
# round(numpy.mean(score)*100,2)

# %%
train = train.drop(["Name", "PassengerId"], axis=1)
test = test.drop(["Name"], axis=1)
train.info()
print("-"*100)
test.info()
# %%
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors=13)
score = cross_val_score(
    clf, train, train["Survived"], cv=k_fold, n_jobs=1, scoring="accuracy")
round(numpy.mean(score)*100, 2)
# %%
# =================================================================
# training 데이터와 training labels를 넣어주고 model을 만드는 단 두 줄의 코딩
# classifier를 불러오는데, 매번 불러올 수 없기 때문에 clf라는 변수에 담아둔다.
clf = KNeighborsClassifier(n_neighbors=13)
clf.fit(train, train["Survived"])
# 이 clf에 fit하게 되면 학습을 해주게 되는데 clf.fit(X_train,Y_train)에서 X_train은 행렬데이터, Y_train은 정답값에 대한 벡터데이터

# =================================================================
# 위의 과정을 거쳐 classifier 학습을 마치고(clf), 학습된 model에 test데이터를 넣고 결과를 예측(predict)하는 코딩
# 그 예측한 값을 prediction 변수로 받는다.
prediction = clf.predict(test)
# %%
submission = pandas.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": prediction})
submission.to_csv("Submission.csv", index=False)
# %%
submission = pandas.read_csv("Submission.csv")
submission.head(10)
# %%
