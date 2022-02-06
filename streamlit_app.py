
import streamlit as st
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.neighbors import NearestNeighbors
import pyodbc
import pymysql

engine = create_engine("mssql+pyodbc://student:student@krtek.prf.jcu.cz/test?driver=ODBC+Driver+17+for+SQL+Server", fast_executemany = True)

@st.cache(allow_output_mutation=True)
def get_connection():
    return create_engine("mssql+pyodbc://student:student@krtek.prf.jcu.cz/test?driver=ODBC+Driver+17+for+SQL+Server", fast_executemany = True)

q1 = 'SELECT * FROM dbo.movie_rating_count_Saad'
q2 = 'SELECT * FROM dbo.movie_title_Saad'
q3 = 'SELECT * FROM dbo.movie_kmeans_Saad'
q4 = 'SELECT * FROM dbo.movie_PCA_Saad'

@st.cache
def read_df1():
  df1 = pd.read_sql_query(q1, get_connection())
  return df1

@st.cache
def read_df2():
  df2 = pd.read_sql_query(q2, get_connection())
  return df2

@st.cache
def read_df3():
  df3 = pd.read_sql_query(q3, get_connection())
  return df3


@st.cache
def read_df4():
  df4 = pd.read_sql_query(q4, get_connection())
  return df4


st.set_page_config(
		page_title= "Movie lens data analysis and recommendations", # String or None. Strings get appended with "• Streamlit".
		 layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
		 #initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
		 #page_icon=None,  # String, anything supported by st.image, or None.
)





col01, col02, col03 = st.columns([1,6,1])

with col01:
	st.write("")

with col02:
	st.title("Welcome to Movie Lens Data Analysis and Recommendations")

with col03:
	st.write("")


col11, col12, col13 = st.columns([3,2,3])

with col11:
	st.write("")

with col12:
	st.subheader("Select Page to View Movies")

with col13:
	st.write("")


rating_df = read_df1()
title_df = read_df2()

#check for rating for by average users for each movie
avg_rating_count_title = rating_df.merge(title_df, on = 'movieId', how='inner')


col21, col22, col23 = st.columns([2,4,2])

with col21:
	st.write("")

with col22:
	n = st.slider("Select Page Number via Slider:", 0, 1185)

with col23:
	st.write("")

user_select = n * 50
user_select1 = user_select + 50
#user_select = np.random.randint(avg_rating_count_title.shape[0], size=n)
show_df1  = avg_rating_count_title.iloc[user_select:user_select1, :]


st.subheader("")

st.subheader("")


col31, col32, col33 = st.columns([1,6,1])

with col31:
	st.write("")

with col32:
	st.write(show_df1)

with col33:
	st.write("")
 
st.subheader("")

st.subheader("")

col41, col42, col43 = st.columns([3,2,3])

with col41:
	st.write("")

with col42:
	st.subheader("Insert Title or part of Title to see list")

with col43:
	st.write("")


col51, col52, col53 = st.columns([3,2,3])

with col51:
	st.write("")

with col52:
	user_input = st.text_input("Enter title", 'toy')

with col53:
	st.write("")
 
st.subheader("")


query1 = "Select movieId, title From dbo.movie_title_Saad where title like '%" + user_input + "%'"
dft2 = pd.read_sql(query1, engine)
search_result = avg_rating_count_title.merge(dft2, on = 'movieId', how='inner')
del search_result['title_y']

col61, col62, col63 = st.columns([2,4,2])

with col61:
	st.write("")

with col62:
	st.write(search_result)

with col63:
	st.write("")

st.subheader("")

genre = ['Adventure', 'Comedy', 'Action', 'Drama', 'Crime', 'Children', 'Mystery', 'Documentary', 'Animation', 'Thriller', 'Horror', 'Fantasy', 'Western', 'Film-Noir', 'Romance', 'War', 'Sci-Fi',
       'Musical', 'IMAX']
avg_rat = avg_rating_count_title[avg_rating_count_title['rating_count']>=423] #dataframe of movie filtered by avg amount of rating on each hopefully around 423
sorted_rat = avg_rat.sort_values(by=['rating_count'], ascending=False).reset_index()
del sorted_rat['index']


col71, col72, col73 = st.columns([3,2,3])

with col71:
	st.write("")

with col72:
	st.subheader('Top 10 Movies by Genre')

with col73:
	st.write("")

st.subheader("")


col81, col82, col83 = st.columns([2,4,2])

with col81:
	st.write("")

with col82:
	st.write('Availble genre are: Adventure, Comedy, Action, Drama, Crime, Children, Mystery, Documentary, Animation, Thriller, Horror, Fantasy, Western, Film-Noir, Romance, War, Sci-Fi, Musical, IMAX')

with col83:
	st.write("")

st.subheader("")


col91, col92, col93 = st.columns([1,6,1])

with col91:
	st.write("")

with col92:
	user_input1 = st.text_input("Enter genre", 'Comedy') #, default_value_goes_here)

with col93:
	st.write("")


query2 =  "Select movieId, title, genres From dbo.movie_title_Saad where genres like '%" + user_input1 + "%'"

dft3 = pd.read_sql(query2, engine)
ddq  = sorted_rat[sorted_rat.movieId.isin(dft3.movieId)]
ddq1 = ddq.sort_values(by=['rating'], ascending=False).reset_index()
del ddq1['index']

ss=ddq1.iloc[:3, 1:4] #adding metrics
ss1 = ss.rating
ss2 = ss.rating_count
ss3 = ss.title

st.subheader("")

col15, col16, col17 = st.columns(3)
col15.metric(str(ss2[0]), ss3[0], str(ss1[0]))
col16.metric(str(ss2[1]), ss3[1], str(ss1[1]))
col17.metric(str(ss2[2]), ss3[2], str(ss1[2]))


st.subheader("")

col101, col102, col103 = st.columns([1,6,1])

with col101:
	st.write("")

with col102:
	st.write(ddq1.head(10))

with col103:
	st.write("")

st.subheader("")

col111, col112, col113 = st.columns([2.5,3,2.5])

with col111:
	st.write("")

with col112:
	st.subheader("User who rates the same movie similarly")

with col113:
	st.write("")
 
st.subheader("")

col121, col122, col123 = st.columns([2,4,2])

with col121:
	st.write("")

with col122:
	user_input2 = st.text_input("Enter userId", 1100) #, default_value_goes_here)

with col123:
	st.write("")
 
kmeans_df = read_df3()
usr = kmeans_df[kmeans_df['userId']==int(user_input2)]
usr = usr.reset_index(drop=True)
usr1 = usr['Group']
usr2 = usr1.values
grp = kmeans_df[kmeans_df['Group']==usr2[0]]
grp = grp.reset_index(drop=True)
j = grp.shape[0]
d = np.random.randint(0, j, size=50)
s = grp.loc[d]
s = s.reset_index(drop=True)
s = s['userId']
s = pd.DataFrame(s)



st.subheader("")

col131, col132, col133 = st.columns([2,4,2])

with col131:
	st.write("")

with col132:
	st.write(s.head(20))

with col133:
	st.write("")
 
PCA_df = read_df4()
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(PCA_df)

st.subheader("")

col161, col162, col163 = st.columns([2,4,2])

with col161:
	st.write("")

with col162:
	st.subheader("Recommed movies to user based on userId")

with col163:
	st.write("")
 


st.subheader("")

col141, col142, col143 = st.columns([2,4,2])

with col141:
	st.write("")

with col142:
	user_input3 = st.text_input("Enter userId for movie recommendation", 1100) #, default_value_goes_here)

with col143:
	st.write("")

inp = PCA_df[PCA_df['userId']==int(user_input3)]
distances, indices = knn_model.kneighbors(inp, n_neighbors=6)
indi1 = indices[:, 1:] #since 1st indices is same as user
k = indi1.tolist()
k = k[0] #list of closest neigbours
j = pd.DataFrame()
for i in range(len(k)):
  p = k[i] #get value of indices
  query = 'SELECT * FROM dbo.movie_rating_Saad where userId = ' + str(p)
  n = pd.read_sql(query, engine)
  j = j.append(n) #append all data
pivot = j.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0) #pivot of users and movies
pp = pivot.reset_index()
k1 = pp.describe()
k3 = k1.iloc[1:2, 1:].values #only the mean row
k4 = k1.columns
k4 = k4[1:]
k5 = pd.DataFrame(k4)
k5['avg_rating'] = k3.T
new5 = title_df.merge(k5, on = 'movieId', how='inner')
del new5['genres']
#getting movies rated by original user 
us1 = indices[:, 0]
us1 = us1[0]
query = 'SELECT * FROM dbo.movie_rating_Saad where userId = ' + str(us1)
df_og = pd.read_sql(query, engine)
mov = df_og['movieId']
not_in = new5[~new5.movieId.isin(mov)]
no = not_in.sort_values(by=['avg_rating'], ascending=False).reset_index()
del no['index']
cols = ['movieId', 'title']
t5 = no[cols] #now write t5 as head for top 10 mov to recommend

col151, col152, col153 = st.columns([2,4,2])

with col151:
	st.write("")

with col152:
	st.write(t5.head(10))

with col153:
	st.write("")