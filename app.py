
import numpy as np
from flask import Flask, request, jsonify, render_template
# from flask_ngrok import run_with_ngrok
import pickle as pk
import pandas as pd
from sklearn import linear_model
import pickle as pk
app = Flask(__name__)
# run_with_ngrok(app)

# #load required data from workspace
# Salesperson_and_location = pd.read_csv("Salesperson and clientid (1).csv")
# Salesperson_rating = pd.read_csv("Salesperson closing ratio (1).csv")
# city_wise_ratio = pd.read_excel("City wise closing ratio salesperson (1).xlsx")
# #load id and city columns from test data
fields = ["id", "City"]

testData = pd.read_csv("test data (1).csv", usecols=fields)
modelfile = 'model.pkl'
model = pk.load(open(modelfile, 'rb'))
@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   '''
   For rendering results on HTML GUI
   '''
   int_features = [int(x) for x in request.form.values()]
   # final_features = [np.array(int_features)]
   
   id = int_features[0]



#    #getting the unique city names
#    cities = Salesperson_and_location.City.unique()

#    #read salespersons names and their overall closing ratio
#    salespersons = Salesperson_rating["SalespersonName"].tolist()
#    Closing_Ratio_list = Salesperson_rating["Closing Ratio"].tolist()

#    # This funciton doubles the space between two names 
#    #to match those in  Salesperson and clientId datafram
#    import re
#    def doubleSpace(name):
#      name = re.sub(' ', '  ', name)
#      return name

#    # save the new names in a separate list
#    spaced_salespersons= []

#    for i, name in enumerate(salespersons):
#      if name != 'Mark  ':
#        spaced_salespersons.append(doubleSpace(salespersons[i]))
#      else: 
#        spaced_salespersons.append('Mark  ')


#    # Create the table
#    data = [] 
#    df = pd.DataFrame(data, columns = ['salespersons_name',
#        'city','count', 'overall rating', 'city rating'])

#    for i, name in enumerate(spaced_salespersons):
#      Closing_Ratio = Closing_Ratio_list[i]
#      for city in cities:
#        count = len(Salesperson_and_location[(Salesperson_and_location['SalespersonName'] == name) & (Salesperson_and_location['City']== city)])
#        cityRating = city_wise_ratio[(city_wise_ratio['SalespersonName'] == salespersons[i])  & (city_wise_ratio['City']== city)]['Closing Ratio']
#        if cityRating.values.shape ==(1,):
#          cityRating = float(cityRating.values)
#        else:
#          cityRating = 0
#        new_row = {'salespersons_name':name,
#                  'city':city, 'count':count, 'overall rating': Closing_Ratio,'city rating': cityRating}
#        #append row to the dataframe
#        df = df.append(new_row, ignore_index=True)

#    df['salespersons_name'] = df['salespersons_name'].astype("category")
#    df['city'] = df['city'].astype("category")
#    df['count'] = df['count'].astype("int")
#    df['overall rating'] = df['overall rating'].astype("float64")
#    df['city rating'] = df['city rating'].astype("float64")
   df = pd.read_csv("df.csv")
   df['salespersons_name'] = df['salespersons_name'].astype("category")
   df['city'] = df['city'].astype("category")
   df['count'] = df['count'].astype("int")
   df['overall rating'] = df['overall rating'].astype("float64")
   df['city rating'] = df['city rating'].astype("float64")


   #Testing


   #Check which city corresponds to input id
   cityName=testData[(testData['id'] == id)]['City']
   cityName=cityName.values[0]

   # Create a dataframe of the sorted salespersons
   df1 = df[(df['city']== cityName)]
   df1 = df1.sort_values(["count", "overall rating"], 
                         ascending = (False, False))

   #Print only highest three
   df2 = df1.iloc[:3]
   df3= df1.iloc[3:]
#    model = linear_model.LinearRegression()

#    model.fit(pd.get_dummies(df3.drop(columns="city rating")), df3["city rating"])


   y_hat = model.predict(pd.get_dummies(df2.drop(columns="city rating")))
   y_hat[y_hat < 0] = 0
   df2['predicted city rating'] = y_hat
   df3 = df2.sort_values(["predicted city rating"], 
                         ascending = (False))

   output = df3['salespersons_name'].values[0]

   return render_template('index.html', prediction_text='salespersons_name: {}'.format(output))


if __name__ == "__main__":
   app.run(debug=True)
