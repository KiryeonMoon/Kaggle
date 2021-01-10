# %%
# 데이터 분석 자료 소환
# 데이터를 씹뜯맛즐 하는 패키지 2개
import pandas as pd
import numpy as np

# 화가가 그림을 그려면 붓, 물감같은 도구가 필요하겠지?
# 시각화 자료 소환
import seaborn as sbn
import matplotlib.pyplot as plt  # 노트북에 자료를 그리기 위해..

# 마지막으로, 머신러닝 툴을 소환해야 함
# 툴은 알고리즘 따라 여러 종류가 있음
# 옛날부터 머신러닝이 발전해오면서 연구자들이 오픈커뮤니티에 쌓아온 자료들
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 훈련 및 테스트 데이터를 불러와야 한다.
# 패널 데이터 분석 도구(pandas)로 불러와 각각 train, test라는 이름의 변수에 집어넣는다.(pandas가 소환식같은 셈)
train = pd.read_csv('C:/Users/Be Irreplaceable/Desktop/titanic/train.csv')
test = pd.read_csv('C:/Users/Be Irreplaceable/Desktop/titanic/test.csv')

# print(train.head(5))  # 데이터 대가리(head)부터 아래로 5개 뽑음
# print("="*150)
# print(test.head(5))
# print("="*150)

'''
맨 왼쪽의 숫자는 인덱스 넘버(항목 번호), passengerID는 승객들을 나열해 번호를 매긴 것
Sex : 성별, Age : 나이, SibSp : 같이 승선한 형제자매 및 배우자, Parch : 부모자녀 같이 탑승
Ticket : 표이름>>중요하지 않음(데이터가 일정하거나 규칙성이 있는 게 아님), fare : 요금>>중요(돈 얼마냈는지 중요)
Cabin : 객실, Pclass : 객실 등급, Embarked
'''

'''
test에는 11개의 column이 있고 train에는 10개가 있다.(생존여부 column이 빠져있음)
train에서 생존자 데이터와 나머지 항목들의 상관관계를 수학적 알고리즘으로 파악한 이후,
train 데이터 세트에 생존자 데이터 세트를 빼고 결과를 알아낼 것이다.
옛날 자료이니만큼 빠진 값이 당연히 있을 수 있고, 이 손실데이터에 대해 데이터 정리 필요
'''

# # 정보 소환
# train.info()
# print("="*100)
# test.info()

# 이제, 자료를 알아보기 위해 칼럼 분류가 필요
# 분류 방법 :
# 손가락으로 셀 수 있는 범위가 나와있는 categorical(범주형 : 세기 쉽고 값을 간단하게 파악 가능)
# 숫자나 범위가 방대한 numeric(어쩔 수 없이 시각화 자료 필요. 누가? 인간이)
# 예를들어, 나이같은 것은 범위가 너무 방대하여 숫자로 세기 힘듦

# # 숫자로 된 데이터 분석
# print(train.describe())
'''
[데이터를 보고 판단할 수 있는 정보들]
1. 총 2224명이 탑승했는데, 훈련 샘플은 891개 → 총 데이터 중 40%정도 제공됨
2. 살았는지 죽었는지 여부(survive)는 1(생존)과 0(사망)으로 구분된다.
3. 타이타닉 실제 생존율은 32%, describe 항목에 제공된 survived의 mean값(0.38)으로 봤을 때 훈련세트의 평균 생존율은 38%
   → 실제 생존율이 아닌 훈련 데이터의 생존율이라도 실제값과 유사한, 나름 양질의 데이터
4. Parch(부모 및 아이와 함께 탑승) 데이터를 보면 0~75% 사람들은 0으로 나온다?
   → 0 = 혼자 탐. 즉, 혼자 탄 사람들이 75% 이상이다.
5. 마찬가지로 SibSp(형제자매 및 배우자와 함께 탑승) 테이터를 보면 75% 이상의 값은 1인데, Max값에 몰려있다.
   → 즉, 형제자매 및 배우자와 함께 탄 사람들은 상위 25% 언저리에 몰려있음
6. 나이는 평균 29.6세, 근데 Max는 80세 노인. 즉, 대부분 젊은이들이 탔는데 나이가 지긋하신 분들도 탔다는 의미정도
7. Pclass : 3등 선실에 탄 사람이 제일 많다.(50% 이상?)
'''

# 문자로 된 데이터 분석
# print(train.describe(include="O"))
'''
1. 데이터 갯수가 891로 동일한 걸 봐서 동명이인이 없는 걸 알 수 있다. 다행. 동명이인이 여러명이면 골치아파
2. 이 사고가 난 당시는 성별이 딱 2개.(unique값이 2라는 건, 데이터가 딱 2개로 나뉘었다는 의미)
3. 그 성별 중 제일 많은 건 남성으로 577명이다.(top이 male이고 freq는 577)
4. 객실(Cabin) 값을 눈여겨 볼 필요가 있겠다.
    일반적으로 모든 승객이 1인실에 묵지는 않는다. 친구들이 가든 여럿이서 돈을 모으든 같은 객실을 여러 사람이 공유하면
    그 사람들의 객실 번호는 동일하다. 이러면 객실 이름이 중복되면서 그만큼 unique(서로 다른 값의 갯수)의 숫자로 포착되지 않는다.
    즉, 대체적으로 승객들이 한 객실에 같이 있었다는 의미. 돈을 많이 안 낸 상황이라고 추론 가능?
5. 승선지(Embarked)는 딱 3곳으로 나뉘고(unique값이 3), S에서 탄 사람들이 644명으로 제일 많다.(top값이 s, freq값이 644)
'''

'''
여기까지 데이터를 분석했으면, 데이터가 무슨 의미인지 어느 정도 파악이 됐다고 봐도 무방
결과값(survived)을 더 잘 예측하기 위해 데이터 엔지니어링(데이터를 자르고 붙이고 없애고 수정하는 일련의 작업) 수행

주어진 항목들과, 우리가 구하고자 하는 예측값의 상관관계를 알아보자
중요하지 않아보이는 항목들도 계산을 거치면 중요해질 수도 있다.

관찰하면서 분석까지 어느 정도 같이 진행했다.
이제 그걸 응용해서 가정을 세울 것이다.

생각) 돈을 많이 낸 사람들은 그만큼 많이 살았을까?
구체화) 돈을 많이 낸 사람(Fare값이 높은 개체)과 객실타입(Pclass)에 따른 생존률이 나뉘는지 파악해도 비슷한 결과가 나올까?
        생존율은 평균값(mean)이니까 mean 적용
'''

# Pclass와 Survived를 소환할건데/Pclass로 그룹핑해서(groupby로 묶어서) 소환할거고
# 평균값 보여줄건데/그 평균값은 생존에 따라서 솎아내고 오름차순으로 해줘
# print(train[["Pclass", "Survived"]].groupby(["Pclass"],
#                                             as_index=False).mean().sort_values(by="Survived", ascending=False))
# print("-"*100)
# 1등 선실(돈 많이 낸 사람들)의 생존률이 높다.
# 여기서 이거 값만 보고 "여윽시 돈 많이 내면 먼저 살렸네, 돈 적네 낸 만큼 푸대접을 받았네, 처우가 안좋았네" 이러면 안 된다.
# 우리는 숫자만 보고 결과에 대한 객관적인 예측만 할 뿐이다. 이 예측에 대한 인문학적인 조사는 전문가가 하는 거다.
# 선체 설계구조적으로 3등 선실이 인명 구조가 쉽지 않은 경우였을 수도 있고
# 당시 미국 이민법에 따라 외지인, 외국인을 3등 선실에 격리시켰는데 이런 점때문에 말이 안통했던 것이 영향을 끼쳤을 수도 있고
# 좌우지간 수치적인 결과 팩트("선실 등급이 높을 수록 생존률이 높다.")만 인지하고 여기다 의견 얹지 말 것

# 성별에 따라 생존율 파악
# # 성별에 따라/생존률
# print(train[["Sex", "Survived"]].groupby(
#     ["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False))

# SibSp와 Survived 비교
# print(train[["SibSp", "Survived"]].groupby("SibSp", as_index=False).mean())
# 가족 한 명과 같이 탄 사람들이 가장 생존률이 높았다.

# 분석한 데이터를 근거로 시각화 ㄱㄱ
# 생존자와 사망자를 (Survived) 왼/오른쪽으로 나누어 도표 두 개 그리기
graph = sbn.FacetGrid(train, col="Survived")  # FaceGrid는 도표 그리는 함수
graph.map(plt.hist, "Age")

'''
히스토그램(막대도형) 그래프는 연속적인 숫자 데이터(나이 등)를 파악하는 데 큰 도움이 된다.
이런 데이터를 파악해서 우리가 가정해본 패턴이 정확한 지 파악할 수 있다.
예를 들어서, 탑승객 중 유아들 생존율이 과연 높은가?라는 질문에 우리가 일일이 파악하고자 하는
모든 데이터 항목을 들여다 본다 치면 엄청난 노가다이지만, 히스토그램을 그려봄으로서
갓난애기들이 사망자 대비 생존자가 많다는 걸 한 눈에 알 수 있다.
20세 언저리 사람들이 많이 사망했음 또한 알 수 있다.
생존여부를 떠나 대다수 승객들이 15-35세에 몰려있음을 알 수 있다.
>> 이런 시각화 자료를 보면 이 다음에 어떤 작업을 수행해야 하는지 방향이 나온다.

[이 시각화 자료를 통해 알 수 있는 정보들]
머신을 설정하고 만들 때 어떤 변수를 집어넣어야 깔끔하게 결과가 나오는지 알 수 있음
나이가 중요한 변수임을 확인함
통계상 공백들은 평균값을 넣어서 계산해주면 수월하다.
그리고 나이는 연속형 데이터(진짜로 아날로그라는 게 아니고, 분포가 1~80까지 너무 많고
카운팅이 힘든 경우를 다 지칭하는 거)라서, group으로 묶어서 판단하는데
그 때 grouping을 어떻게 하면 좋을 지 판단 가능
'''

# 아까 가정한 것처럼, 3등 선실 사람들이 1,2등 선실 사람들보다 더 많이 사망한 걸 알 수 있다.
# 그렇다면, 각 선실별로 탑승객들의 나이가 다 다를텐데, 여기서 나누어 생각한다.
# 각 1,2,3등 선실을 나누고 / graph 안에서 나이로 나누고 / 생존여부를 나누고 이렇게 유형별로 나눔
grid = sbn.FacetGrid(train, col="Survived", row="Pclass")
grid.map(plt.hist, "Age", bins=15)
# 단순히 graph높이만으로도, 그만큼 사람들이 어디에 많이 탔는 지 알 수 있음.
# 봤더니 3등선실에 많이 탔고, 그 3등선실 중 대부분이 survived=0 그래프에 있음 즉,
# 3등 선실이 숫자도 많고 사망자도 많음
# 결론 : Pclass는 머신러닝에 넣을 중요한 지표가 된다.
# (머신에 넣을 자료에서 제외하지 않기로 결정, 남기기로 결정)


# 다음 고민
# 승선지 및 선실 등급별 남자와 여자가 있는데, 남자와 여자의 생존율이 각 등급별/객실별 선실에 따라
# 과연 다를까?
# 그렇다면 이 질문에 답하기 위해서 그래프를 그려 또 다시 확인할 수 있을 것 같다.
# 위와 마찬가지로 도표 그려보자
# 남자와 여자는 딱 둘로 나뉘므로 막대그래프(histogram) 말고
# 가시성 좋은 쩜도표를 사용ㄱㄱ
# 가로를 선실 등급별로 나누고, 그래프 안에서 남/녀 생존율을 그린 다음 파악 시도
grid = sbn.FacetGrid(train, row="Embarked")  # 그래프의 프레임을 그림
grid.map(sbn.pointplot, "Pclass", "Survived", "Sex",  # x축에 선실등급, y축에 생존률, 구분을 성별 두 개로.
         pallette="bright")  # 그래프 프레임 안에 내용을 채우는 것
grid.add_legend()  # 범례(legend) 넣기
# 여기서 알 수 있는 점 : C에서 탑승한 승객만 남자가 여자보다 생존율이 높다.
# 무조건 C에서 탑승했기 때문에 생존율이 높은 게 아니고,
# 객실 등급과 승선지의 관계, 객실 등급과 생존율 사이의 관계가 겹쳐졌기 때문에
# 이런 결과가 나왔을 수도 있다.(might) 단정지을 수 없다!
# 그러니까, 무조건 C에서 승선했기 때문에 생존율이 낮다? No 알 수 없다.

# 그리고 아까 3등 선실에 머문 승객들의 생존률이 낮다고 했는데,
# 3등 선실에 탑승한 사람들일지라도 이 사람들이 C나 Q에서 탑승했을 경우에 생존률이 1, 2등 선실보다 더 높았다.
# >> 승선지에 따라 3등 선실 승객들에서 보이는 다양한 생존율이 관측된다.

# 결론 : 당연하지만, 가설(낸 돈과 생존률의 비례)을 세우고 검증이 완료되었으니
# Sex 항목도 모델에 넣기로 결정

# 질문 : 아니 남녀는 당연히 넣어야 하는 거 아님? >> 우린 이과니까 가설을 세우고 검증하는 과정이 필요.
# 마찬가지로, 승선지도 넣기로 결정(승선지에 따라서 결과값이 다른 게 보이니까 영향을 주는 변수로 볼 수 있음)

# 시각화의 마지막으로, 범주형 데이터와 숫자형 데이터의 상관관계를 살펴본다.
# 질문 : 귀찮게 왜 살펴보냐? >> 데이터가 그렇게 주어졌으니까
# 쉽게 말해, 승선지와 성별은 범주형데이터(손가락으로 셀 수 있다는 뜻)
# 승선지, 성별 <> 요금(연속형 데이터, 손가락으로 셀 수 없는 넓은 범위 데이터) <> 생존여부(손가락으로 셀 수 있고 숫자로 되어있음)

# 질문 : 그럼 어떻게 할 것인가?
# 그래프를 여러 개 그려서 좌측과 우측을 생존자로 나누고
# 위에서부터 승선지를 그리면 2x3 여섯 개 그래프 프레임이 그려짐
# 그 안에 세부적으로 성별에 따른 요금 평균을 구하면 된다.
# 성별에 따른 요금 평균을 구하면 딱 2개로 나뉜다(평균값이니까) 이걸 뚱뚱한 막대기로 그릴꺼니까 barplot 적용

grid = sbn.FacetGrid(train, col="Survived",
                     row="Embarked")  # 2ㅌ3 6개 그래프 큰 틀을 짬

# x축에 성별, y축에 요금. 신뢰구간(ci)은 그래프에 안그리겠다.
grid.map(sbn.barplot, "Sex", "Fare", ci=None)
# 결과적으로, Q행선지 빼고 요금을 많이 낸 사람들이 더 많이 살았다.(세개 중 두 개 일치하니 일반적)
# >> 이거 중요! 머신러닝 모델을 설정하고 훈련시킬 때 넣을 자료에 꼭 포함시켜야겠다는 판단이 선다.
# 그리고 의외로 승선지도 생존율에 중요한 영향을 끼치는 것으로 보인다는 판단.
# (C에서 탄 사람들이 많이 살았고, 산 사람들을 봤더니 S에서 탔다.이것 역시 중요 요소라 포함시킬거다.)
# Fare 항목 데이터값이 다양 >> 이걸 band(범위)로 묶어서 데이터로 쓰면 좋을 것 같다.

'''
정리
우리가 시각화해서 살펴본 자료랑, 그 전에 groupby로 살펴본 자료에 따라서,
머신러닝 모델을 만들고 훈련시킬 데이터를 정제할 때 포함시킬 항목들을 선택하고,
그것이 진짜로 상관관계가 있는지 확인했다.

머신러닝에 넣을 항목 다시 확인
1. 요금(묶어서)
2. 성별
3. 객실등급(1,2,3)
4. 승선지
5. 나이(묶어서)
6. SibSp, Parch(묶어서)

이제, 데이터를 자르고 붙여서 정제한다.
'''
train.head(5)
# 이름 중요구별부 빼고 다 지우고, 성별도 남녀 글자를 0,1 숫자로 변경하고 나이는 범위별로 묶고
# 티켓 버리고 요금도 band로 묶고 Cabin도 버리고 Embarked도 알파벳을 숫자로 바꾸고
# 이제부터 데이터 자르고 붙이고 정제

# 티켓과 Cabin을 훈련/테스트 데이터에서 아예 없앰(drop)
train = train.drop(["Ticket", "Cabin"], axis=1)
test = test.drop(["Ticket", "Cabin"], axis=1)

combine = [train, test]

train.info()
test.info()

train.describe(include="O")
# Name의 count수와 unique수가 같은 걸 보니 동명이인은 없어보이지만, 타이틀로만 묶는다면 banding이 될 수도!

combine = [train, test]

# 이름 외 title이라는 새로운 column 생성
# dataset에 있는 Name 항목의 String(문제 데이터)를 Extract(추출)해라. 근데 괄호()안에 있는 방식으로 해라.
for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract("([A-Za-z]+)\.", expand=False)

train.head(5)

pd.crosstab(train["Title"], train["Sex"])

pd.crosstab(train["Title"], train["Survived"])
# 결과가 잘 나오니 타이틀 추출은 성공.
# 이렇게 타이틀을 추출해서 훈련 데이터로 남겨서 쓰는 게 예측값(생존여부)를 파악하는 데 큰 도움이 될 것 같음.
# 이 자료를 쓰겠다.

for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(
        ["Lady.", "Countess.", "Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer."], "Rare.")
    dataset["Title"] = dataset["Title"].replace("Mlle.", "Miss.")
    dataset["Title"] = dataset["Title"].replace("Ms.", "Miss.")
    dataset["Title"] = dataset["Title"].replace("Mme.", "Mrs.")
# 이름을 다 바꾸었으니, 이제 이름은 손가락으로 셀 수 있는 범주형 데이터로 볼 수 있겠지?

# 그룹바이로 타이틀 별 생존률 본다.
train[["Title", "Survived"]].groupby(["Title"], as_index=False).mean()


# 문자를 숫자로 바꾸는 작업=맵핑 수행
title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare.": 5}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)  # 빠진 값들은 0으로 바꾸라는 의미

train.head(20)

# 이제 안전하게 trian/test 데이터에서 이름(Name)을 빼버린다.
# 또한 PassengerId도 더 이상 필요 없음. 데이터 자체적인 고유 특징을 이미 다 부여했기 때문에
# 데이터 구분을 위해 초기에 입력한 승객번호는 더 이상 필요가 없음
train = train.drop(["Name", "PassengerId"], axis=1)
test = test.drop(["Name"], axis=1)

combine = [train, test]
# test에 있는 PassengerId는 지우는 거 아님

train.head(10)

for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1}).astype(int)
train.head(10)

# 나이(Age)와 운임(Fare)은 연속된 숫자값
'''
이제 슬슬 빠져있는 값들을 채워넣으면 될 것 같다.
방법1. 가장 쉬움. 평균과 표준편차 사이의 Random값을 넣는 방법
'''

# %%
