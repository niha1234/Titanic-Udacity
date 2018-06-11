
# Titanic Data Analyses for Survival Chance factors

# Introduction
The titanic data set taken from kaggle data science website, features 891 entries as sample dataset. We will do basic analyses of dataset inculding graphs, function to determine that what are factors in order to know the survival chances of a passengers who boarded the ship.

### Data Dictionary

Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex	
Age	Age in years	
sibsp	# of siblings / spouses aboard the Titanic	
parch	# of parents / children aboard the Titanic	
ticket	Ticket number	
fare	Passenger fare	
cabin	Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

### Variable Notes

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

### Data Mining Questions:

1. Does age group increases the survival chances?
2. Does being female or male in in particular age group increases the survival chance?
3. Does passenger class increases the the survival chance?



```python
import pandas as pd
import numpy as np

# Read data
titanic_data = pd.read_csv('titanic_data.csv')
```


```python
titanic_data.head()  # Initail analyses with getting info on data such names and what kind of
                     # values they consist in the dataframe
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_data.info() # Data set information related to data type and count which gives more insigts about
                    # null values in each variable
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    

With the information on titanic data set we can see that total entries are 891, however Age field has 714 non null entries, Cabin field has 204 non null entries, and Embarked has 889 non null entries. 
This suggests that either these information were missing or not entered. Whichever the case, we need to either substitute with a new entry accordingly with the field or remove it from the dataset.


```python
titanic_data.describe() # Describes general statistics of each variable
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



Max age is 80 years, and minimum age is .42 years. Simlarly max fare is 512.32, and minimum fare is 0.00.

# Data Cleaning

Below is the Number of passengers boarded the ship from each Emabrked point in the given sample data set


```python
Count_Embarked = titanic_data.groupby('Embarked').size() 
Count_Embarked
```




    Embarked
    C    168
    Q     77
    S    644
    dtype: int64



It is clear that most of the passengers had departed from SouthHampton which is denoted by 'S' in the data set. We can make assumption that null entries in the Embarked field can be replaced with SouthHampton with the dontion 'S'.


```python
titanic_data['Embarked'] = titanic_data.Embarked.replace(np.NaN, 'S')
Count_Embarked = titanic_data.groupby('Embarked').size()
Count_Embarked

```




    Embarked
    C    168
    Q     77
    S    646
    dtype: int64



Binning Age and creating a new column with age_group 


```python

bins = [0,10,20,30,40,50,60,70,80,90,100]
age_labels = ['Kid', 'Teen', '20s', '30s', '40s', '50s','60s','70s','80s','90s']
titanic_data['age_group'] = pd.cut(titanic_data.Age, bins, right=False, labels=age_labels)

titanic_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>20s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>30s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>20s</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>30s</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>30s</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_data_survivors_before_cleaning = titanic_data.groupby('Survived').get_group(1)

titaninc_data_non_survivors_before_cleaning = titanic_data.groupby('Survived').get_group(0)

```


```python
age_group_survivors_before_cleaning = titanic_data_survivors_before_cleaning.groupby('age_group').size()
```


```python
age_group_non_survivors_before_cleaning = titaninc_data_non_survivors_before_cleaning.groupby('age_group').size()
```


```python
age_mean = titanic_data['Age'].mean()
print(age_mean)
titanic_data['Age'] = titanic_data.Age.replace(np.NaN, age_mean)
titanic_data['age_group'] = titanic_data.age_group.replace(np.NaN, '20s')

titanic_data.info()
```

    29.69911764705882
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            891 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       891 non-null object
    age_group      891 non-null category
    dtypes: category(1), float64(2), int64(5), object(5)
    memory usage: 84.6+ KB
    


```python
#remove unwanted column
titanic_data_new = titanic_data.drop(['PassengerId','Name','Cabin'],1)
titanic_data_new.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>S</td>
      <td>20s</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>30s</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>S</td>
      <td>20s</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>S</td>
      <td>30s</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>S</td>
      <td>30s</td>
    </tr>
  </tbody>
</table>
</div>



# Data Exploration 

### Grouping titanic sample data frame in Survivors and Non Survivors


```python
## Grouping survivors and non survivors passengers in two variables
titanic_data_survivors_after_cleaning = titanic_data_new.groupby('Survived').get_group(1)

titanic_data_non_survivors_after_cleaning = titanic_data_new.groupby('Survived').get_group(0)


```


```python
age_group_survivors_after_cleaning = titanic_data_survivors_after_cleaning.groupby('age_group').size()
age_group_non_survivors_after_cleaning = titanic_data_non_survivors_after_cleaning.groupby('age_group').size()


```


```python

import matplotlib.pyplot as plt
%pylab inline 
import seaborn as  sns

plt.figure(figsize=(10,5))

N = 10
index = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(index, age_group_survivors_before_cleaning, width, color='#d62728')
p2 = plt.bar(index+width, age_group_survivors_after_cleaning, width)

plt.ylabel('Survival')
plt.title('Scores by age_group')
plt.xticks(index, ('Kids', 'Teens', '20s', '30s', '40s','50s','60s','70s','80s','90s','100s'))
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Before Cleaning', 'After Cleaning'))



plt.show()
```

    Populating the interactive namespace from numpy and matplotlib
    

    C:\Users\NIHA\Anaconda3\lib\site-packages\IPython\core\magics\pylab.py:161: UserWarning: pylab import has clobbered these variables: ['axes']
    `%matplotlib` prevents importing * from pylab and numpy
      "\n`%matplotlib` prevents importing * from pylab and numpy"
    


![png](output_24_2.png)



```python

plt.figure(figsize=(10,5))

N = 10
index = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(index, age_group_non_survivors_before_cleaning, width, color='#d62728')
p2 = plt.bar(index+width, age_group_non_survivors_after_cleaning, width)

plt.ylabel('Non Survival')
plt.title('Scores by age_group')
plt.xticks(index, ('Kids', 'Teens', '20s', '30s', '40s','50s','60s','70s','80s','90s','100s'))
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Before Cleaning', 'After Cleaning'))



plt.show()
```


![png](output_25_0.png)


We can see that count of survivors and non survivors, both have increased in age group 20s as we replaced null entries in age with 20s. However, the count of 20s age group was already higher than the other age group in non survivors dataset. The count before cleaning with 20s and 30s were almost same, whereas after cleaning and replacing value with the mean age widened the gap between 20s and 30s age group. We have to assume that replacing age with mean age for null entries wouldn't affect our analysis. 

The next step includes surviviors and non survivors by age group to determine if there is any impact on survival rate due to the age group.



```python
## Calculating percent of survivors and non survivors age group
titanic_data_survivors_age_group = titanic_data_survivors_after_cleaning.groupby('age_group').size()
titanic_data_non_survivors_age_group = titanic_data_non_survivors_after_cleaning.groupby('age_group').size()


total_survivors_age_group = titanic_data_survivors_age_group + titanic_data_non_survivors_age_group
percent_survivor_age_group = (titanic_data_survivors_age_group*1.0/total_survivors_age_group)*100
percent_non_survivor_age_group = (titanic_data_non_survivors_age_group*1.0/total_survivors_age_group)*100
```


```python
plt.figure(figsize=(10,5))

N = 10
index = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(index, percent_survivor_age_group, width, color='#d62728')
p2 = plt.bar(index+width, percent_non_survivor_age_group, width)

plt.ylabel('Survival Percent')
plt.title('Scores by age_group')
plt.xticks(index, ('Kids', 'Teens', '20s', '30s', '40s','50s','60s','70s','80s','90s','100s'))
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Survivors', 'Non Survivors'))



plt.show()
```


![png](output_29_0.png)


By looking at bar chart above for survivora and non survivors by age group, Kids survival rate is highest, followed by 80s. It is interesting that 80s age group has 100% survival rate. We would like to know in this sample data set how many passengers were in 80s group



```python
print(titanic_data_survivors_age_group)
print(titanic_data_non_survivors_age_group)
```

    age_group
    Kid      38
    Teen     41
    20s     129
    30s      73
    40s      34
    50s      20
    60s       6
    70s       0
    80s       1
    90s       0
    dtype: int64
    age_group
    Kid      24
    Teen     61
    20s     268
    30s      94
    40s      55
    50s      28
    60s      13
    70s       6
    80s       0
    90s       0
    dtype: int64
    

In this sample dataset 80s group has only one passenger in survivors group and zero passengers in non survivors.WHereas 70s age group has zero percent survival rate. Therefore,we can not determine that whether old age people had more help or people who cannot be saved were not able reach to lifeboats. Or it can be assumed they let young people go first because of thier age.

Grouping sample data frame of survivors and non survivors by Gender


```python
titanic_data_survivors_female = titanic_data_survivors_after_cleaning.groupby('Sex').get_group('female')
titanic_data_survivors_male = titanic_data_survivors_after_cleaning.groupby('Sex').get_group('male')
titanic_data_non_survivors_female = titanic_data_non_survivors_after_cleaning.groupby('Sex').get_group('female')
titanic_data_non_survivors_male = titanic_data_non_survivors_after_cleaning.groupby('Sex').get_group('male')

survivors_female_age_group = titanic_data_survivors_female.groupby('age_group').size()
survivors_male_age_group = titanic_data_survivors_male.groupby('age_group').size()
non_survivors_female_age_group = titanic_data_non_survivors_female.groupby('age_group').size()
non_survivors_male_age_group = titanic_data_non_survivors_male.groupby('age_group').size()

total_survivors_age_group = survivors_female_age_group + survivors_male_age_group
percent_female_age_group = (survivors_female_age_group*1.0/total_survivors_age_group)*100

percent_male_age_group = (survivors_male_age_group*1.0/total_survivors_age_group)*100

```


```python
plt.figure(figsize=(10,5))

N = 10
index = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(index, percent_female_age_group, width, color='#d62728')
p2 = plt.bar(index+width, percent_male_age_group, width)

plt.ylabel('Survival Percent')
plt.title('Scores by age_group')
plt.xticks(index, ('Kids', 'Teens', '20s', '30s', '40s','50s','60s','70s','80s','90s','100s'))
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Female', 'Male'))



plt.show()
```


![png](output_35_0.png)


Survival rate of female and male is same in Kids, whereas in all other age groups survival rate of female is more than male. It can be assumed except Kids age group, in all other age groups they followed the women first policy.

## Grouping sample frame of survivors and non survivors by Passenger Class


```python
titanic_data_survivors_Pclass = titanic_data_survivors_after_cleaning.groupby('Pclass').size()
titanic_data_non_survivors_Pclass = titanic_data_non_survivors_after_cleaning.groupby('Pclass').size()


total_passengers_Pclass = titanic_data_survivors_Pclass +titanic_data_non_survivors_Pclass
percent_survivors_Pclass = (titanic_data_survivors_Pclass*1.0/total_passengers_Pclass)*100
percent_non_survivors_Pclass = (titanic_data_non_survivors_Pclass*1.0/total_passengers_Pclass)*100

```


```python
plt.figure(figsize=(10,5))

N = 3
index = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(index, percent_survivors_Pclass, width, color='#d62728')
p2 = plt.bar(index+width, percent_non_survivors_Pclass, width,
             )

plt.ylabel('Survival Percent')
plt.title('Scores by Class')
plt.xticks(index, ('Class 1', 'Class 2', 'Class 3'))
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Survivors', 'Non Survivors'))



plt.show()
```


![png](output_39_0.png)


Looking at the above graph, it is clear that class 1 passengers had more chances of survivals. 

# Conclusion

Above graph suggests that 
1. Age group affected the survival chances. Percent survival rate is highest in Kids Age group. However, all other group almost close to each other. With respect to kids we can say that age group affected the survival chances of passengers 

2. Gender also was a key variable in order to have a more survival rate. Female gender had more survival rate than the Male survival. Addintinally, Teens age group had better survival rate than any other age group. It may be possible they were easy to find out, whereas kids group mostly stayed with their parents or guardian and couldn't reach to lifeboats quickly.

3. Similarly, Class 1 had least number of casualities, whereas Class 3 had most number of casualities. This also suggests that class one passengers had more life boats and equipments closer to thier cabins which helped them to reach to safe places quicker. Whereas class two and class three passengers didn't have enough life boats and equipments.

# Limitations

I have imputed null values for age with the mean value of age which sort of represent limitations, because it increases 20s age group count dramatically. Therefore, all the results are presented in percent form so that increase in count of passengers in 20s group won't affect conclusion dramatically.

## References: 

1.  https://stackoverflow.com/questions/31726643/how-do-i-get-multiple-subplots-in-matplotlib

2. https://www.kaggle.com/c/titanic/data

3. http://matplotlib.org/examples/pylab_examples/bar_stacked.html

4. http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html#bar-plots

5. https://en.wikipedia.org/wiki/Women_and_children_first



```python

```
