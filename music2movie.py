################################################
# End-to-End Music2movie Machine Learning Pipeline I
################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.data
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from datetime import datetime



#config.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv('DATA/movies_metadata.csv', low_memory=False)


################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe[num_cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

    def cat_summary(dataframe, col_name, plot=False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    def num_summary(dataframe, numerical_col, plot=False):
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            dataframe[numerical_col].hist(bins=20)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show(block=True)

    def target_summary_with_num(dataframe, target, numerical_col):
        print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

    def target_summary_with_cat(dataframe, target, categorical_col):
        print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


grab_col_names(df)
#Output
#(['adult', 'status', 'video'], --cat_cols
# ['revenue', 'runtime', 'vote_average', 'vote_count'], --num_cols
# ['belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 'poster_path', 'production_companies', 'production_countries', 'release_date', 'spoken_languages', 'tagline', 'title']) --cat_but_car

# Distinguishing between categorical and numerical variables
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

#Dataset overview
check_df(df)
##################### Shape #####################
#(45466, 24)
##################### Types #####################
#adult                     object
# belongs_to_collection     object
# budget                    object
# genres                    object
# homepage                  object
# id                        object
# imdb_id                   object
# original_language         object
# original_title            object
# overview                  object
# popularity                object
# poster_path               object
# production_companies      object
# production_countries      object
# release_date              object
# revenue                  float64
# runtime                  float64
# spoken_languages          object
# status                    object
# tagline                   object
# title                     object
# video                     object
# vote_average             float64
# vote_count               float64
# dtype: object
##################### Head #####################
# adult                              belongs_to_collection    budget                                             genres                              homepage     id    imdb_id original_language               original_title                                           overview popularity                       poster_path                               production_companies                               production_countries release_date       revenue  runtime                                   spoken_languages    status                                            tagline                        title  video  vote_average  vote_count
# 0  False  {'id': 10194, 'name': 'Toy Story Collection', ...  30000000  [{'id': 16, 'name': 'Animation'}, {'id': 35, '...  http://toystory.disney.com/toy-story    862  tt0114709                en                    Toy Story  Led by Woody, Andy's toys live happily in his ...  21.946943  /rhIRbceoE9lR4veEXuwCC2wARtG.jpg     [{'name': 'Pixar Animation Studios', 'id': 3}]  [{'iso_3166_1': 'US', 'name': 'United States o...   1995-10-30 373554033.000   81.000           [{'iso_639_1': 'en', 'name': 'English'}]  Released                                                NaN                    Toy Story  False         7.700    5415.000
# 1  False                                                NaN  65000000  [{'id': 12, 'name': 'Adventure'}, {'id': 14, '...                                   NaN   8844  tt0113497                en                      Jumanji  When siblings Judy and Peter discover an encha...  17.015539  /vzmL6fP7aPKNKPRTFnZmiUfciyV.jpg  [{'name': 'TriStar Pictures', 'id': 559}, {'na...  [{'iso_3166_1': 'US', 'name': 'United States o...   1995-12-15 262797249.000  104.000  [{'iso_639_1': 'en', 'name': 'English'}, {'iso...  Released          Roll the dice and unleash the excitement!                      Jumanji  False         6.900    2413.000
# 2  False  {'id': 119050, 'name': 'Grumpy Old Men Collect...         0  [{'id': 10749, 'name': 'Romance'}, {'id': 35, ...                                   NaN  15602  tt0113228                en             Grumpier Old Men  A family wedding reignites the ancient feud be...    11.7129  /6ksm1sjKMFLbO7UY2i6G1ju9SML.jpg  [{'name': 'Warner Bros.', 'id': 6194}, {'name'...  [{'iso_3166_1': 'US', 'name': 'United States o...   1995-12-22         0.000  101.000           [{'iso_639_1': 'en', 'name': 'English'}]  Released  Still Yelling. Still Fighting. Still Ready for...             Grumpier Old Men  False         6.500      92.000
# 3  False                                                NaN  16000000  [{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...                                   NaN  31357  tt0114885                en            Waiting to Exhale  Cheated on, mistreated and stepped on, the wom...   3.859495  /16XOMpEaLWkrcPqSQqhTmeJuqQl.jpg  [{'name': 'Twentieth Century Fox Film Corporat...  [{'iso_3166_1': 'US', 'name': 'United States o...   1995-12-22  81452156.000  127.000           [{'iso_639_1': 'en', 'name': 'English'}]  Released  Friends are the people who let you be yourself...            Waiting to Exhale  False         6.100      34.000
# 4  False  {'id': 96871, 'name': 'Father of the Bride Col...         0                     [{'id': 35, 'name': 'Comedy'}]                                   NaN  11862  tt0113041                en  Father of the Bride Part II  Just when George Banks has recovered from his ...   8.387519  /e64sOI48hQXyru7naBFyssKFxVd.jpg  [{'name': 'Sandollar Productions', 'id': 5842}...  [{'iso_3166_1': 'US', 'name': 'United States o...   1995-02-10  76578911.000  106.000           [{'iso_639_1': 'en', 'name': 'English'}]  Released  Just When His World Is Back To Normal... He's ...  Father of the Bride Part II  False         5.700     173.000
#################### Tail #####################
# adult belongs_to_collection budget                                             genres                              homepage      id    imdb_id original_language       original_title                                           overview popularity                       poster_path                               production_companies                               production_countries release_date  revenue  runtime                          spoken_languages    status                                     tagline                title  video  vote_average  vote_count
# 45461  False                   NaN      0  [{'id': 18, 'name': 'Drama'}, {'id': 10751, 'n...  http://www.imdb.com/title/tt6209470/  439050  tt6209470                fa              رگ خواب        Rising and falling between a man and woman.   0.072051  /jldsYflnId4tTWPx8es3uzsB1I8.jpg                                                 []             [{'iso_3166_1': 'IR', 'name': 'Iran'}]          NaN    0.000   90.000    [{'iso_639_1': 'fa', 'name': 'فارسی'}]  Released  Rising and falling between a man and woman               Subdue  False         4.000       1.000
# 45462  False                   NaN      0                      [{'id': 18, 'name': 'Drama'}]                                   NaN  111109  tt2028550                tl  Siglo ng Pagluluwal  An artist struggles to finish his work while a...   0.178241  /xZkmxsNmYXJbKVsTRLLx3pqGHx7.jpg             [{'name': 'Sine Olivia', 'id': 19653}]      [{'iso_3166_1': 'PH', 'name': 'Philippines'}]   2011-11-17    0.000  360.000         [{'iso_639_1': 'tl', 'name': ''}]  Released                                         NaN  Century of Birthing  False         9.000       3.000
# 45463  False                   NaN      0  [{'id': 28, 'name': 'Action'}, {'id': 18, 'nam...                                   NaN   67758  tt0303758                en             Betrayal  When one of her hits goes wrong, a professiona...   0.903007  /d5bX92nDsISNhu3ZT69uHwmfCGw.jpg  [{'name': 'American World Pictures', 'id': 6165}]  [{'iso_3166_1': 'US', 'name': 'United States o...   2003-08-01    0.000   90.000  [{'iso_639_1': 'en', 'name': 'English'}]  Released                      A deadly game of wits.             Betrayal  False         3.800       6.000
# 45464  False                   NaN      0                                                 []                                   NaN  227506  tt0008536                en  Satana likuyushchiy  In a small town live two brothers, one a minis...   0.003503  /aorBPO7ak8e8iJKT5OcqYxU3jlK.jpg               [{'name': 'Yermoliev', 'id': 88753}]           [{'iso_3166_1': 'RU', 'name': 'Russia'}]   1917-10-21    0.000   87.000                                        []  Released                                         NaN     Satan Triumphant  False         0.000       0.000
# 45465  False                   NaN      0                                                 []                                   NaN  461257  tt6980792                en             Queerama  50 years after decriminalisation of homosexual...   0.163015  /s5UkZt6NTsrS7ZF0Rh8nzupRlIU.jpg                                                 []   [{'iso_3166_1': 'GB', 'name': 'United Kingdom'}]   2017-06-09    0.000   75.000  [{'iso_639_1': 'en', 'name': 'English'}]  Released                                         NaN             Queerama  False         0.000       0.000
#################### NA #####################
# adult                        0
# belongs_to_collection    40972
# budget                       0
# genres                       0
# homepage                 37684
# id                           0
# imdb_id                     17
# original_language           11
# original_title               0
# overview                   954
# popularity                   5
# poster_path                386
# production_companies         3
# production_countries         3
# release_date                87
# revenue                      6
# runtime                    263
# spoken_languages             6
# status                      87
# tagline                  25054
# title                        6
# video                        6
# vote_average                 6
# vote_count                   6
# dtype: int64
#################### Quantiles #####################
# 0.000  0.050  0.500        0.950         0.990          1.000
# revenue       0.000  0.000  0.000 47808918.500 273087551.660 2787965087.000
# runtime       0.000 11.000 95.000      138.000       185.000       1256.000
# vote_average  0.000  0.000  6.000        7.800         8.700         10.000
# vote_count    0.000  0.000 10.000      434.000      2183.820      14075.000


# Analyzing categorical variables
for col in cat_cols:
    cat_summary(df, col)

#                                                    adult  Ratio
# adult
# False                                               45454 99.974
# True                                                    9  0.020
#  - Written by Ørnås                                     1  0.002
#  Rune Balot goes to a casino connected to the O...      1  0.002
#  Avalanche Sharks tells the story of a bikini c...      1  0.002
#########################################
# status  Ratio
# status
# Released          45014 99.006
# Rumored             230  0.506
# Post Production      98  0.216
# In Production        20  0.044
# Planned              15  0.033
# Canceled              2  0.004
#########################################
# video  Ratio
# video
# False  45367 99.782
# True      93  0.205
#########################################

correlation_matrix(df, num_cols)
#vote_average and revenue: 0.084

#vote_average and runtime: 0.16

#vote_average and vote_count: 0.12

#revenue and vote_count: 0.81

#runtime and revenue: 0.10

#runtime and vote_count: 0.11

#Copy of dataset

df_ = df.copy()

#Elimination of irrelevant variables
list1 = df[df['genres'].str.contains('animation', case=False,
                                     na=False)].index  # "Since this study is intended for adults, I am removing films of this type."
df = df.drop(list1)

columns_to_drop = [
    'adult', 'belongs_to_collection', 'homepage', 'production_companies',
    'production_countries', 'id', 'imdb_id', 'poster_path', 'budget', 'video',
    'status', 'original_language', 'spoken_languages', 'genres', 'tagline',
    'title', 'popularity', 'release_date', 'revenue',
]

df = df.drop(columns=columns_to_drop)
df.dropna(inplace=True)
df.isnull().sum()
#original_title    0
#overview          0
#runtime           0
#vote_average      0
#vote_count        0
#dtype: int64
df.shape #(42587, 5)


################################################
# NLP Analysis Section Using VADER and NRC
################################################
nltk.download('punkt', download_dir='C:/Users/Leyla/nltk_data')
nltk.download('punkt_tab')
nltk.data.path.append('C:/Users/Leyla/nltk_data')

nltk.download('punkt')
nltk.download('stopwords')

df_cl = df.copy() #As a precaution against potential errors

df['overview'] = df['overview'].apply(lambda x: x.lower())
df['overview'] = df['overview'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation))) #Punctuation was removed
df['overview'] = df['overview'].apply(lambda x: word_tokenize(x)) #Sentences were split into individual words
stop_words = set(stopwords.words('english')) # Created a set of stopwords including articles and other words with little standalone meaning
df['overview'] = df['overview'].apply(lambda x: [word for word in x if word not in stop_words]) #Stopwords were removed
nrc = pd.read_csv("DATA/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
                  names=["word", "emotion", "association"],
                  sep="\t")
# A dictionary was created in dataframe format, similar to the 'names=' argument,
# mapping words to their corresponding emotions along with numerical values.
# The result is as follows:
#      word       emotion  association
#0    aback         anger            0
#1    aback  anticipation            0
#2    aback       disgust            0
#3    aback          fear            0
#4    aback           joy            0
#5    aback      negative            0
#6    aback      positive            0
#7    aback       sadness            0
#8    aback      surprise            0
#9    aback         trust            0
#10  abacus         anger            0
#11  abacus  anticipation            0

nrc = nrc[nrc['association'] == 1] # Filters words that have emotional significance

emotion_dict = defaultdict(list) #It automatically creates an empty list as a default value for any key that doesn’t exist yet.
#This makes it easier to append items to keys without manually checking if the key exists.

for index, row in nrc.iterrows():
    emotion_dict[row['word']].append(row['emotion']) # Assigns the corresponding emotions to each word

emotion_list = ['anger', 'anticipation', 'disgust', 'fear',
                'joy', 'sadness', 'surprise', 'trust',
                'positive', 'negative']

# Adds 1 to the corresponding emotion if a matching word is found in the tokenized description
def get_emotion_counts(token_list):
    counts = dict.fromkeys(emotion_list, 0)
    for word in token_list:
        if word in emotion_dict:
            for emo in emotion_dict[word]:
                counts[emo] += 1
    return counts

#   anger  anticipation  disgust  fear  joy  sadness  surprise  trust  positive  negative
#1      1             5        1     3    5        1         3      4         5         2
#2      4             3        2     5    0        3         3      1         2         8
#3      0             1        0     0    1        0         3      1         3         1
#4      0             4        0     1    0        0         0      1         1         0
#5      3             0        1     4    0        2         2      2         5         4

emotion_df = df['overview'].apply(get_emotion_counts).apply(pd.Series)

df = pd.concat([df, emotion_df], axis=1) #The relevant emotion_df is joined to the main dataframe

# Valence Aware Dictionary and sEntiment Reasoner is a rule-based sentiment analysis model.
# It provides the positive, negative, and neutral sentiment ratios of sentences, as well as the overall emotional tone (compound score).
# compound ≥ 0.05 → positive
# compound ≤ -0.05 → negative
# -0.05 < compound < 0.05 → neutral

# VADER nesnesini oluştur
analyzer = SentimentIntensityAnalyzer() # Initializing the SentimentIntensityAnalyzer class from the VADER library

df[['neg', 'neu', 'pos', 'compound']] = df['overview'].apply(
    lambda x: pd.Series(analyzer.polarity_scores(str(x))) #Polarity scores are added to the dataframe.
)


def emotion_coef(data): #"Here, coefficients were assigned based on the weights obtained from the context of the description.
    compound = data['compound']

    if compound >= 0.05:
        a, b = -1, +2
    elif compound <= -0.05:
        a, b = -2, +1
    else:
        a, b = -0.5, +0.5
    score = (
            a * (data['negative'] + data['anger'] + data['fear'] + data['sadness'] + data['disgust']) +
            b * (data['anticipation'] + data['positive'] + data['joy'] + data['trust'] + data['surprise'])
    )
    return score

df['emotion_score'] = df.apply(emotion_coef, axis=1)



################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    print(f"Column: {col_name}, Type: {dataframe[col_name].dtype}")
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    condition = (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    return condition.any()
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df["emotion_group"] = pd.qcut(df["emotion_score"], q=6) #kategorik değişken oluşturduk, burada elimizde olan emotion_score'u silmiyoruz henüz
# emotion_group
#(-16.0, -5.0]        7511
#(-113.001, -16.0]    7446
#(0.5, 7.0]           7241
#(19.0, 140.0]        7082
#(7.0, 19.0]          6933
#(-5.0, 0.5]          6374


# Positiveness: the weighted combination of positive emotions
df['positiveness'] = (
    df["joy"] * 1.0 +
    df["trust"] * 1.0 +
    df["positive"] * 1.0 +
    df["pos"] * 10.0) #because the 'pos' value is usually very small

# # Negativeness: the combination of negative emotions
df["negativeness"] = (
    df["anger"] * 1.0 +
    df["fear"] * 1.0 +
    df["sadness"] * 1.0 +
    df["disgust"] * 1.0 +
    df["negative"] * 1.0 +
    df["neg"] * 10.0  # Since the VADER score is small, multiplying it makes sense
)

df["emotional_polarity"] = df["positiveness"] - df["negativeness"]

def genres_by_emotion(score): #Here, we will categorize emotion score ranges based on music genres.
    if score > 19:
        return 'edm, dance, pop, dance pop, electronic, k-pop, j-pop, turkish pop, flamenco, house, drum and bass'
    elif score > 7:
        return 'funk, romantic, alternative pop, emo rap, permanent wave, art pop, sertanejo, r&b, arabic pop, latin, latin pop, arabesk'
    elif score > 0.5:
        return 'lo-fi, sad indie, jazz, tango, bossa nova,indian classical, chill, country, afrobeat, reggae, indie, singer-songwriter, melodic rap, baroque pop, indie pop, chillhop, canadian hip hop'
    elif score > -5:
        return 'classical, blues, ambient, alternative rock, soul, dark academia, sleep'
    elif score > -16:
        return 'hard rock, trap, post-punk, trance'
    else:
        return 'metal, Dark Trap, Experimental, german techno'


#Assigned genre groups based on the emotion_score

df["genre_group"] = df["emotion_score"].apply(genres_by_emotion)

# Since emotion_group represented numeric ranges, new categorical representations were created to preserve its structure
manual_encoding = {
    "(-113.001, -16.0]": 0,
    "(-16.0, -5.0]": 1,
    "(-5.0, 0.5]": 2,
    "(0.5, 7.0]": 3,
    "(7.0, 19.0]": 4,
    "(19.0, 140.0]": 5
}

df["emotion_group_encoded"] = df["emotion_group"].astype(str).map(manual_encoding)
df["emotion_group_encoded"] = df["emotion_group_encoded"].astype("object")



df_encoded = df.copy() #As a precaution against potential errors

#Since we won't be using the 'overview' variable, we can now remove it.

df = df.drop('overview', axis=1)
df = df.drop('original_title', axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)


#Categorical variables were encoded
df = one_hot_encoder(df, cat_cols, drop_first=True)
df = df.astype({col: int for col in df.select_dtypes(include='bool').columns})

#Since the contents of the numerical variables were converted to strings, I am applying the following operations.

numeric_cols = ['runtime','anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust', 'pos', 'emotional_polarity', 'positiveness', 'negativeness',"vote_average", "vote_count", "anger", "joy", "positive", "neg", "neu", "compound", "emotion_score"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Removing the 'emotion_group' variable as it is no longer needed


df = df.drop('emotion_group', axis=1)



X = df.drop('emotion_score', axis=1)

y = df["emotion_score"]

######################################################
# 3. Model
######################################################

def base_models(X, y, scoring="r2"):
    print("Base Models....")

    regressors = [
        ('LR', LinearRegression()),
        ('KNN', KNeighborsRegressor()),
        ('RF', RandomForestRegressor()),
        ('GBM', GradientBoostingRegressor()),
        ('XGB', XGBRegressor()),

    ]

    for name, reg in regressors:
        cv_results = cross_validate(reg, X, y, cv=3, scoring=scoring)
        score = cv_results['test_score'].mean()
        print(f"{name:<8} | {scoring}: {round(score, 4)}")


base_models(X, y, scoring="r2")
#Base Models....
#LR       | r2: 0.9452
#KNN      | r2: 0.816
#RF       | r2: 0.9958
#GBM      | r2: 0.9917
#XGB      | r2: 0.9976

regressors = [
    ('LR', LinearRegression()),
    ('KNN', KNeighborsRegressor()),
    ('RF', RandomForestRegressor()),
    ('GBM', GradientBoostingRegressor()),
    ('XGB', XGBRegressor()),
]

def evaluate_models(X, y, regressors, cv=5):
    results = []

    for name, model in regressors:
        # Negatif RMSE
        rmse_scores = -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
        # Negatif MAE
        mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
        # R2
        r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv)

        results.append({
            "Model": name,
            "R2 Mean": np.mean(r2_scores).round(4),
            "RMSE Mean": np.mean(rmse_scores).round(4),
            "MAE Mean": np.mean(mae_scores).round(4)
        })

    return pd.DataFrame(results)



evaluate_models(X, y, regressors)
#  Model  R2 Mean  RMSE Mean  MAE Mean
#0    LR    0.946      4.823     3.437
#1   KNN    0.825      8.623     5.297
#2    RF    0.996      1.277     0.839
#3   GBM    0.992      1.879     1.459
#4   XGB    0.998      0.947     0.609

#These columns were removed to prevent data leakage,
# ensuring the model generalizes better and reflects real-world prediction scenarios

leakage_cols = [
    'compound', 'positive', 'negative', 'pos', 'neg', 'neu',
    'positiveness', 'negativeness', 'emotional_polarity'
]

X_clean = X.drop(leakage_cols, axis=1)
base_models(X_clean, y, scoring="r2")

#LR       | r2: 0.9172
# KNN      | r2: 0.6611
# RF       | r2: 0.9756
# GBM      | r2: 0.9724
# XGB      | r2: 0.978


cross_val_score(RandomForestRegressor(), X_clean, y, cv=5, scoring='r2')
#array([0.97611569, 0.97593207, 0.97652382, 0.97627515, 0.97578511])

#I initially intended to combine the models in an ensemble,
# but due to numerous errors, I decided to proceed with the RandomForestRegressor,
# which had the highest performance and was the most robust against overfitting.

X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.05, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print(comparison.head(30))
#    Actual  Predicted
# 0   20.000     24.970
# 1  -13.000    -10.010
# 2  -12.000    -11.300
# 3   -7.000     -7.720
# 4   -2.000     -1.680
# 5   68.000     63.400
# 6   18.000     15.930
# 7  -56.000    -55.980
# 8  -12.000    -12.030
# 9   -2.000     -2.130
# 10  22.000     22.210
# 11  -9.000     -8.320
# 12  19.000     14.780
# 13   0.000      0.125
# 14  -7.000     -7.370
# 15  -0.500     -2.780
# 16  14.000     15.170
# 17   7.000      3.870
# 18 -52.000    -45.080
# 19  30.000     22.540
# 20 -24.000    -24.060
# 21  38.000     35.000
# 22   3.000      3.465
# 23 -12.000     -8.210
# 24 -25.000    -22.890
# 25  20.000     22.460
# 26  12.000     14.620
# 27  -5.000    -11.050
# 28  20.000     20.880
# 29  -3.000     -1.500

#Feature importance
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)
print(importances.head(10))

#genre_group_metal, Dark Trap, Experimental, german techno                                                                              0.277
# emotion_group_encoded_5                                                                                                                0.245
# genre_group_edm, dance, pop, dance pop, electronic, k-pop, j-pop, turkish pop, flamenco, house, drum and bass                          0.235
# joy                                                                                                                                    0.043
# genre_group_funk, romantic, alternative pop, emo rap, permanent wave, art pop, sertanejo, r&b, arabic pop, latin, latin pop, arabesk   0.037
# emotion_group_encoded_4                                                                                                                0.034
# fear                                                                                                                                   0.023
# anger                                                                                                                                  0.022
# genre_group_hard rock, trap, post-punk, trance                                                                                         0.019
# trust


plt.figure(figsize=(12,6))
importances.head(20).plot(kind='bar')
plt.title("Özellik Önemleri")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.show()


importances = importances[importances > 0.02]



param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_clean, y)
#Fitting 5 folds for each of 20 candidates, totalling 100 fits
#RandomizedSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                   # n_iter=20, n_jobs=-1,
                   # param_distributions={'max_depth': [5, 10, 15, 20, None],
                   #                      'min_samples_leaf': [1, 2, 4],
                   #                      'min_samples_split': [2, 5, 10],
                   #                      'n_estimators': [100, 200, 300, 500]},
                   # random_state=42, scoring='neg_root_mean_squared_error',
                   # verbose=1)


print("Best RMSE:", -random_search.best_score_)
print("Best Parameters:", random_search.best_params_)

#Best RMSE: 3.1631552432154897
#Best Parameters: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': None}

best_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42
)

best_model.fit(X_clean, y)
scores = cross_val_score(best_model, X_clean, y, cv=5, scoring='neg_root_mean_squared_error')
scores1 = cross_val_score(best_model, X_clean, y, cv=5, scoring='r2')
#array([-3.03096627, -3.09865246, -3.18109359, -3.26373931, -3.24132459])

#Save the model as a .pkl (pickle) file.
joblib.dump(best_model, "DATA/emotion_score_model.pkl")




#Here, we load the dataset from which we will recommend movies

movie_df = pd.read_csv('DATA/movie_imdb.csv')

movie_df.to_csv('DATA/movie_df_ml.csv')
movie_df_c =movie_df.copy()

# movie_df final_popularity feature türetimi

movie_df['final_score'] = movie_df['vote_count_score'] * movie_df['vote_average']
movie_df['final_popularity'] = movie_df['popularity'] * movie_df['final_score']

movie_df["release_date"] = pd.to_datetime(movie_df["release_date"], errors="coerce")

tarih = datetime(2021, 1, 18)

movie_df['day_ago'] = (tarih - movie_df['release_date']).dt.days

def recommend_varied_films(genre_keyword, X_columns, model, movie_df,
                           tolerance=3.0, top_n=3, candidate_pool=15):
    """
    genre_keyword : Spotify'dan gelen müzik türü (örn. 'deep house')
    X_columns     : Modelin eğitildiği feature isimleri (X_clean.columns)
    model         : emotion_score tahmini yapan eğitilmiş model
    movie_df      : Filmlerin bulunduğu dataframe (title, emotion_score, final_score vs.)
    tolerance     : Emotion score ± kaç birim aralığında film seçilsin
    top_n         : Kaç film önerilsin
    candidate_pool: Rastgele seçim yapılacak havuzun büyüklüğü
    """

    # 🔮 1. Emotion score tahmini yap
    matched_cols = [col for col in X_columns if genre_keyword.lower() in col.lower()]

    if not matched_cols:
        # ❌ Tür encode edilmiş veri setinde yoksa → fallback: final_popularity'ye göre öner
        pool = movie_df.sort_values(by="final_popularity", ascending=False).head(candidate_pool)
        sampled = pool.sample(n=min(top_n, len(pool)), random_state=None)
        return sampled[["title", "final_popularity", "vote_average", "genre_group"]]

    # 🎯 2. Model input'u oluştur
    input_df = pd.DataFrame(data=[0] * len(X_columns), index=X_columns).T
    input_df.columns = X_columns
    for col in matched_cols:
        input_df.at[0, col] = 1

    # 🔢 3. Tahmini al
    try:
        predicted_score = model.predict(input_df)[0]
    except:
        # Model hata verirse yine fallback'e dön
        pool = movie_df.sort_values(by="final_popularity", ascending=False).head(candidate_pool)
        sampled = pool.sample(n=min(top_n, len(pool)), random_state=None)
        return sampled[["title", "final_popularity", "vote_average", "genre_group"]]

    # 🔍 4. Emotion score'a yakın filmleri filtrele
    mask = movie_df["emotion_score"].between(predicted_score - tolerance, predicted_score + tolerance)
    filtered = movie_df[mask]

    if filtered.empty:
        # 🎬 Uygun film yoksa yine fallback
        pool = movie_df.sort_values(by="final_popularity", ascending=False).head(candidate_pool)
        sampled = pool.sample(n=min(top_n, len(pool)), random_state=None)
        return sampled[["title", "final_popularity", "vote_average", "genre_group"]]

    # 🌟 5. Final_score'a göre sıralayıp rastgele seçim yap
    filtered_sorted = filtered.sort_values(by="final_score", ascending=False).head(candidate_pool)
    sampled = filtered_sorted.sample(n=min(top_n, len(filtered_sorted)), random_state=None)

    return sampled[["title", "emotion_score", "vote_average", "final_score", "genre_group"]]

recommend_varied_films('Experimental', X_clean.columns, best_model, movie_df, top_n=3)

