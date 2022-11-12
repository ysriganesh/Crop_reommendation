import pyttsx3                                                            # Importing pyttsx3 library to convert text into speech.
import pandas as pd                                                       # Importing pandas library
from sklearn import preprocessing                                         # Importing sklearn library. This is a very powerfull library for machine learning. Scikit-learn is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
from sklearn.neighbors import KNeighborsClassifier                        # Importing Knn Classifier from sklearn library.
import numpy as np                                                        # Importing numpy to do stuffs related to arrays
import PySimpleGUI as sg
import webbrowser
import datetime
from pathlib import Path
from scipy.interpolate import make_interp_spline

from docxtpl import DocxTemplate
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

import warnings



data = pd.read_csv('C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/Crop_recommendation.csv')

print(data.head())

document_path = Path(__file__).parent / "C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/vendor-contract.docx"
doc = DocxTemplate(document_path)

today = datetime.datetime.today()
today_in_one_week = today + datetime.timedelta(days=7)




engine = pyttsx3.init('sapi5')                                            # Defining the speech rate, type of voice etc.
voices = engine.getProperty('voices')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-20)
engine.setProperty('voice',voices[0].id)

def speak(audio):                                                         # Defining a speak function. We can call this function when we want to make our program to speak something.
	engine.say(audio)
	engine.runAndWait()


le = preprocessing.LabelEncoder()                                         # Various machine learning algorithms require numerical input data, so you need to represent categorical columns in a numerical column. In order to encode this data, you could map each value to a number. This process is known as label encoding, and sklearn conveniently will do this for you using Label Encoder.
crop = le.fit_transform(list(data["label"]))

NITROGEN = list(data["N"])                                        # Making the whole row consisting of nitrogen values to come into nitrogen.
PHOSPHORUS = list(data["P"])                                    # Making the whole row consisting of phosphorus values to come into phosphorus.
POTASSIUM = list(data["K"])                                      # Making the whole row consisting of potassium values to come into potassium.
TEMPERATURE = list(data["temperature"])                                  # Making the whole row consisting of temperature values to come into temperature.
HUMIDITY = list(data["humidity"])                                        # Making the whole row consisting of humidity values to come into humidity.
PH = list(data["ph"])                                                    # Making the whole row consisting of ph values to come into ph.
RAINFALL = list(data["rainfall"])

features = [NITROGEN, PHOSPHORUS, POTASSIUM, TEMPERATURE, HUMIDITY, PH, RAINFALL]           # Zipping all the features together
features = np.array(features,dtype=float)

features = features.transpose()
print(features.shape)
print(crop.shape)

xdata = data.iloc[:, 0:7].values
ydata = data.iloc[:, 7].values

xtrain, xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.2, random_state=884)
x_st = StandardScaler()
xtrain = x_st.fit_transform(xtrain)
xtest = x_st.fit_transform(xtest)

acc_list = []
err_rate = []

neighbors = np.linspace(1, 50, 50)
neighbors = neighbors.astype(int)

for K in neighbors:
  classifier = KNeighborsClassifier(n_neighbors = K)
  classifier.fit(xtrain, ytrain)
  y_pred = classifier.predict(xtest)

  accuracy = round(acc(ytest, y_pred)*100, 3)

  acc_list.append(accuracy)
  err_rate.append(np.mean(y_pred != ytest))

xy = make_interp_spline(neighbors, acc_list)
xz = make_interp_spline(neighbors, err_rate)
x = np.linspace(1, 50, 1000)
y = xy(x)
z = xz(x)

plt.figure(figsize = (13, 7))
plt.subplot(2, 1, 1)

plt.xlabel('K value')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs K', fontweight = 'bold')
plt.xlim(min(neighbors), max(neighbors))

plt.subplot(2, 1, 2)

plt.xlabel('K value')
plt.ylabel('Loss')
plt.title('Loss vs K', fontweight = 'bold')
plt.xlim(min(neighbors), max(neighbors))



K_opt = acc_list.index(max(acc_list))
print('\nOptimal value of K = ', K_opt)




print('\nOptimal value of K = ', K_opt)

classifier = KNeighborsClassifier(n_neighbors=K_opt+1)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

accuracy = acc(ytest, y_pred)*100
print('Accuracy of the training Model : ', round(accuracy, 3), '%')


model = KNeighborsClassifier(2)
model.fit(features, crop)

speak("SIR I AM HERE TO HELP YOU REGARDING THE CROP RECOMMENDATION  JUST CLICK THE  BUTTONS FOR FILLING DETAILS")

layout = [[sg.Text('                      Crop Recommendation Assistant', font=("roman", 30), text_color = 'yellow')],  # Defining the layout of the Graphical User Interface. It consist of some text, Buttons, and blanks to take Input.
		  [sg.Text('Hello Farmers I am here to Assist you with Crop Recommendation and Loan Prediction', font=("roman", 20))],

		  [sg.Text(' In Crop Recommendation i will try to recommend the suitable crop accroding to the condtion given by user                                :', font=("roman", 20))],
		  [sg.Text(' To Help larger number of farmers i am designed in very simple way click the options and make use of me                          :', font=("roman", 20))],

		  [sg.Button('CROP PREDCTION', font=("roman", 20)), sg.Button('Print Statement', font=("roman", 20)),sg.Button('Quit', font=("roman", 20))],
		  [sg.Text('Help??', justification='center', expand_x=True, font=('Courier New', 20, 'underline'), enable_events=True, key='LINK2')]]
window = sg.Window('Crop Recommendation Assistant', layout)



while True:
	event, values = window.read()

	if event == 'Quit' or sg.WINDOW_CLOSED:
		break
	elif event in ('LINK1', 'LINK2'):
		import webbrowser

		webbrowser.open("http://localhost:8501/")
		continue

	elif event == 'Print Statement':

		layout = [
			[sg.Text("CLIENT:"), sg.Input(key="CLIENT", do_not_clear=False)],
			[sg.Text("place name:"), sg.Input(key="PLACE", do_not_clear=False)],
			[sg.Text("Acres:"), sg.Input(key="LAND", do_not_clear=False)],
			[sg.Text("Date:"), sg.Input(key="DATE", do_not_clear=False)],
			[sg.Text("NITROGEN :"), sg.Input(key="NITROGEN", do_not_clear=False)],
			[sg.Text("PHOSPOROUS :"), sg.Input(key="PHOSPOROUS", do_not_clear=False)],
			[sg.Text("POTASSIUM :"), sg.Input(key="POTASSIUM", do_not_clear=False)],
			[sg.Text("TEMPARATURE :"), sg.Input(key="TEMPARATURE", do_not_clear=False)],
			[sg.Text("HUMIDITY :"), sg.Input(key="HUMIDITY", do_not_clear=False)],
			[sg.Text("PH VALUE  :"), sg.Input(key="PHVALUE ", do_not_clear=False)],
			[sg.Text("RAINFALL:"), sg.Input(key="RAINFALL", do_not_clear=False)],

			[sg.Button("Create Contract"), sg.Exit()],
		]

		window = sg.Window("Contract Generator", layout, element_justification="right")

		while True:
			event, values = window.read()
			if event == sg.WIN_CLOSED or event == "Exit":
				exit()
			if event == "Create Contract":


				# Render the template, save new word document & inform user
				doc.render(values)
				output_path = Path(__file__).parent / f"{values['CLIENT']}-contract.docx"
				doc.save(output_path)
				sg.popup("File saved", f"File has been saved here: {output_path}")

		window.close()





	elif event == 'CROP PREDCTION':
		speak("Welcome to Crop Recommendation Assistant")
		speak("Remember to fill the details in only interger formate")
		layout = [[sg.Text('                      Crop Recommendation Assistant', font=("Helvetica", 30),
						   text_color='yellow')],
				  # Defining the layout of the Graphical User Interface. It consist of some text, Buttons, and blanks to take Input.
				  [sg.Text('Please enter the following details :-', font=("Helvetica", 20))],
				  # We have defined the text size, font type, font size, blank size, colour of the text in the GUI.
				  [sg.Text('Enter ratio of Nitrogen in the soil                                  :',font=("Helvetica", 20)), sg.Input(font=("Helvetica", 20), size=(20, 1))],
				  [sg.Text('Enter ratio of Phosphorous in the soil                           :',font=("Helvetica", 20)), sg.Input(font=("Helvetica", 20), size=(20, 1))],
				  [sg.Text('Enter ratio of Potassium in the soil                               :',font=("Helvetica", 20)), sg.Input(font=("Helvetica", 20), size=(20, 1))],
				  [sg.Text('Enter average Temperature value around the field        :', font=("Helvetica", 20)),
				   sg.Input(font=("Helvetica", 20), size=(20, 1)), sg.Text('*C', font=("Helvetica", 20))],
				  [sg.Text('Enter average percentage of Humidity around the field :', font=("Helvetica", 20)),
				   sg.Input(font=("Helvetica", 20), size=(20, 1)), sg.Text('%', font=("Helvetica", 20))],
				  [sg.Text('Enter PH value of the soil                                            :',font=("Helvetica", 20)), sg.Input(font=("Helvetica", 20), size=(20, 1))],
				  [sg.Text('Enter average amount of Rainfall around the field        :', font=("Helvetica", 20)),
				   sg.Input(font=("Helvetica", 20), size=(20, 1)), sg.Text('mm', font=("Helvetica", 20))],
				  [sg.Text(size=(50, 1), font=("Helvetica", 20), text_color='yellow', key='-OUTPUT1-')],
				  [sg.Button('Submit', font=("Helvetica", 20)), sg.Button('Quit', font=("Helvetica", 20))]]
		window = sg.Window('Crop Recommendation Assistant', layout)

		while True:
			event, values = window.read()
			if event == sg.WINDOW_CLOSED or event == 'Quit':  # If the user will press the quit button then the program will end up.
				exit()
			print(values[0])
			nitrogen_content = values[0]  # Taking input from the user about nitrogen content in the soil.
			phosphorus_content = values[1]  # Taking input from the user about phosphorus content in the soil.
			potassium_content = values[2]  # Taking input from the user about potassium content in the soil.
			temperature_content = values[3]  # Taking input from the user about the surrounding temperature.
			humidity_content = values[4]  # Taking input from the user about the surrounding humidity.
			ph_content = values[5]  # Taking input from the user about the ph level of the soil.
			rainfall = values[6]
			predict1 = np.array([nitrogen_content, phosphorus_content, potassium_content, temperature_content, humidity_content,ph_content, rainfall], dtype=float)
			print(predict1)
			predict1 = predict1.reshape(1, -1)
			print(predict1)
			predict1 = model.predict(predict1)
			print(predict1)
			crop_name = str()
			if predict1 == 0:  # Above we have converted the crop names into numerical form, so that we can apply the machine learning model easily. Now we have to again change the numerical values into names of crop so that we can print it when required.
				crop_name = 'Apple'
			elif predict1 == 1:
				crop_name = 'Banana'
			elif predict1 == 2:
				crop_name = 'Blackgram'
			elif predict1 == 3:
				crop_name = 'Chickpea'
			elif predict1 == 4:
				crop_name = 'Coconut'
			elif predict1 == 5:
				crop_name = 'Coffee'
			elif predict1 == 6:
				crop_name = 'Cotton'
			elif predict1 == 7:
				crop_name = 'Grapes'
			elif predict1 == 8:
				crop_name = 'Jute'
			elif predict1 == 9:
				crop_name = 'Kidneybeans'
			elif predict1 == 10:
				crop_name = 'Lentil'
			elif predict1 == 11:
				crop_name = 'Maize'
			elif predict1 == 12:
				crop_name = 'Mango'
			elif predict1 == 13:
				crop_name = 'Mothbeans'
			elif predict1 == 14:
				crop_name = 'Mungbeans'
			elif predict1 == 15:
				crop_name = 'Muskmelon'
			elif predict1 == 16:
				crop_name = 'Orange'
			elif predict1 == 17:
				crop_name = 'Papaya'
			elif predict1 == 18:
				crop_name = 'Pigeonpeas'
			elif predict1 == 19:
				crop_name = 'Pomegranate'
			elif predict1 == 20:
				crop_name = 'Rice'
			elif predict1 == 21:
				crop_name = 'Watermelon'

			if int(humidity_content) >= 1 and int(
					humidity_content) <= 33:  # Here I have divided the humidity values into three categories i.e low humid, medium humid, high humid.
				humidity_level = 'low humid'
			elif int(humidity_content) >= 34 and int(humidity_content) <= 66:
				humidity_level = 'medium humid'
			else:
				humidity_level = 'high humid'

			if int(temperature_content) >= 0 and int(
					temperature_content) <= 6:  # Here I have divided the temperature values into three categories i.e cool, warm, hot.
				temperature_level = 'cool'
			elif int(temperature_content) >= 7 and int(temperature_content) <= 25:
				temperature_level = 'warm'
			else:
				temperature_level = 'hot'

			if int(rainfall) >= 1 and int(
					rainfall) <= 100:  # Here I have divided the humidity values into three categories i.e less, moderate, heavy rain.
				rainfall_level = 'less'
			elif int(rainfall) >= 101 and int(rainfall) <= 200:
				rainfall_level = 'moderate'
			elif int(rainfall) >= 201:
				rainfall_level = 'heavy rain'

			if int(nitrogen_content) >= 1 and int(
					nitrogen_content) <= 50:  # Here I have divided the nitrogen values into three categories.
				nitrogen_level = 'less'
			elif int(nitrogen_content) >= 51 and int(nitrogen_content) <= 100:
				nitrogen_level = 'not to less but also not to high'
			elif int(nitrogen_content) >= 101:
				nitrogen_level = 'high'

			if int(phosphorus_content) >= 1 and int(
					phosphorus_content) <= 50:  # Here I have divided the phosphorus values into three categories.
				phosphorus_level = 'less'
			elif int(phosphorus_content) >= 51 and int(phosphorus_content) <= 100:
				phosphorus_level = 'not to less but also not to high'
			elif int(phosphorus_content) >= 101:
				phosphorus_level = 'high'

			if int(potassium_content) >= 1 and int(
					potassium_content) <= 50:  # Here I have divided the potassium values into three categories.
				potassium_level = 'less'
			elif int(potassium_content) >= 51 and int(potassium_content) <= 100:
				potassium_level = 'not to less but also not to high'
			elif int(potassium_content) >= 101:
				potassium_level = 'high'

			if float(ph_content) >= 0 and float(
					ph_content) <= 5:  # Here I have divided the ph values into three categories.
				phlevel = 'acidic'
			elif float(ph_content) >= 6 and float(ph_content) <= 8:
				phlevel = 'neutral'
			elif float(ph_content) >= 9 and float(ph_content) <= 14:
				phlevel = 'alkaline'

			print(crop_name)
			print(humidity_level)
			print(temperature_level)
			print(rainfall_level)
			print(nitrogen_level)
			print(phosphorus_level)
			print(potassium_level)
			print(phlevel)

			speak("Sir according to the data that you provided to me. The ratio of nitrogen in the soil is  " + nitrogen_level + ". The ratio of phosphorus in the soil is  " + phosphorus_level + ". The ratio of potassium in the soil is  " + potassium_level + ". The temperature level around the field is  " + temperature_level + ". The humidity level around the field is  " + humidity_level + ". The ph type of the soil is"+phlevel+" . The amount of rainfall is  " + rainfall_level)  # Making our program to speak about the data that it has received about the crop in front of the user.
			window['-OUTPUT1-'].update(
				'The best crop that you can grow : ' + crop_name)  # Suggesting the best crop after prediction.
			speak("The best crop that you can grow is  " + crop_name)  # Speaking the name of the predicted crop.

		window.close()
