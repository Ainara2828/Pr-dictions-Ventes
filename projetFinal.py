import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
import numpy as np
from xlwt import Workbook
from tkinter import *
from functools import partial

#93 articles et 35 semaines

Var = pd.read_csv("DonneesVariance.csv") #/data/Documents/Cours EISTI/ING2/Semestre 1/ProjetRegLin/
Moy = pd.read_csv("DonneesMoyenne.csv")
EcTy = pd.read_csv("DonneesEcartType.csv")

########################## DETERMINATION PREDICTION, ERREURS ET MODELE OPTMIAL ####################################

def predict(x,reg):
   return reg[0] * x + reg[1]

def regression(x,y):
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	return slope,intercept, r_value, p_value, std_err


def calculErreur(liste):
	somme=0
	for i in range(0,len(liste)-1):
		if ((np.isinf(liste[i])==False) and (np.isnan(liste[i])==False)):
			somme = liste[i]+somme
	return somme/len(liste)

resVar = []
resMoy = []
resEcTy = []

errVar = []
errMoy = []
errEcTy = []

PvalueVar = []

i=0
while (i<191):
	YVar = Var.iloc[0:len(Var)-2,i]
	XVar = Var.iloc[0:len(Var)-2,i+1]

	YMoy = Moy.iloc[0:len(Moy)-2,i]
	XMoy = Moy.iloc[0:len(Moy)-2,i+1]

	YEcTy = EcTy.iloc[0:len(EcTy)-2,i]
	XEcTy = EcTy.iloc[0:len(EcTy)-2,i+1]

	regVar = regression(XVar,YVar)
	regMoy = regression(XMoy,YMoy)
	regEcTy = regression(XEcTy,YEcTy)

	predicVar = predict(XVar,regVar)
	predicMoy = predict(XMoy,regMoy)
	predicEcTy = predict(XEcTy,regEcTy)

	errVar.append(regVar[4]) #Erreur de regression des 93 articles
	errMoy.append(regMoy[4])
	errEcTy.append(regEcTy[4])

	PvalueVar.append(regVar[3]) #Pvalue des 93 articles

	resVar.append(predicVar) #Prediction des 93 articles
	resMoy.append(predicMoy)
	resEcTy.append(predicEcTy)

	i=i+2


ErreurVariance = "Erreur de régression expliquée par la variance :" + str(calculErreur(errVar)) #la plus faible
ErreurEcTy = "Erreur de régression expliquée par l'écart-type :" + str(calculErreur(errEcTy))
ErreurMoyenne = "Erreur de régression expliquée par la moyenne:" + str(calculErreur(errMoy))

############## GENERER LE GRAPHIQUE ##############################

def genererGraphique(indice):
	X = Var.iloc[0:len(Var)-2,indice+1]
	Y = Var.iloc[0:len(Var)-2,indice]
	plt.scatter(X,Y)
	regr = linear_model.LinearRegression()
	regr.fit(X[:,np.newaxis], Y)
	x_test = np.linspace(np.min(X), np.max(X), 100)
	plt.plot(x_test, regr.predict(x_test[:,np.newaxis]), color='blue', linewidth=3)
	plt.show()

############################################ CHOIX UTILISATEUR ################################################## 

listeArticles = [89005907,89007507,89010978,89011016,89011048,89011119,89011129,89011448,89011642,89011704,89011745,89011747,89012333,89012486,89012516,89074636,89075417,89075967,89077501,89078230,89079659,89090152,89094273,89095030,89504648,89011098,89057825,90005288,90005942,90007068,90010141,90011903,90012743,90013323,90015258,90017500,90020568,90022088,92000110,92000299,92000362,92000381,92000386,92000694,92000741,92000797,92000812,92000813,92000834,92000882,92000951,92000952,92000963,92000965,92000983,
92001063,92001184,92001201,92001232,92001236,92001324,92001341,92001450,92001463,92001468,92001473,92001575,92001726,92001830,92001889,92001944,92001946,92002033,92002072,92002113,92002114,92002117,92002141,92002267,92002347,92002506,92002630,92002636,92002798,92002907,92002916,92002990,92003013,92003033,92003061,92003062,92003112,92003123,92003132,92003161,92003175]

w = Tk()

labelErrVar = Label(w,text=ErreurVariance)
labelErrMoy = Label(w,text=ErreurMoyenne)
labelErrEcTy = Label(w,text=ErreurEcTy)
labelIntro = Label(w,text="Prévisions de la régréssion linéaire par la variance :",font='Helvetica 18 bold')

labelErrVar.grid(row=0,column=0)
labelErrMoy.grid(row=1,column=0)
labelErrEcTy.grid(row=2,column=0)
labelIntro.grid(row=3,column=0)




#PREDICTIONS PAR ARTICLES

#affichage prediction sur les 35 semaines, la p value et l'erreur de l'article

# création listbox articles
lbx = Listbox(w,exportselection=0)
for i in range(0,len(listeArticles)-1):
	lbx.insert(i, listeArticles[i])

lbx.grid(row=4, column=0)


indice = StringVar()
selected_item = StringVar()

def affichagePrevisionArticle():
	lbx.select_set(lbx.curselection()[0])
	indice = lbx.curselection()[0]

	labelResVar = Label(w, text=resVar[int(indice)])
	labelResVar.grid()
	texte = "P-value :" + str(PvalueVar[int(indice)]) + ";  Erreur" + str(errVar[int(indice)])
	labelPred = Label(w, text=texte)
	labelPred.grid()

	#graphique de l'article
	genererGraphique(int(indice))

bt = Button(w, text="Entrer Article", command=affichagePrevisionArticle)
bt.grid(row=5, column=0)


def telechargerArticle():
	articleListe = []
	indice = lbx.curselection()[0]
	book = Workbook() #enregistrement aussi dans un fichier excel
	feuil1 = book.add_sheet('feuille 1')
	articleListe = pd.Series.tolist(resVar[int(indice)])
	for i in range(0,len(articleListe)-1):
		feuil1.write(i,0,articleListe[i])
	book.save('predictionsParArticle.xls')

bt5 = Button(w, text="Télécharger", command=telechargerArticle)
bt5.grid(row=6, column=0)

#PREDICTIONS PAR SEMAINES 

llPredic = []

for i in range(0,len(resVar)-1):
	llPredic.append(pd.Series.tolist(resVar[i]))

lbx2 = Listbox(w,exportselection=0)
indice2 = StringVar()

for i in range(0,36):
	lbx2.insert(i, i)

lbx2.grid(row=4, column=2)

predicSemaine =[]

def affichagePrevisionSemaine():
    lbx2.select_set(lbx2.curselection()[0])
    indice2 = lbx2.curselection()[0]
    for i in range(0,len(llPredic)-1):
    	predicSemaine.append(llPredic[i][int(indice2)])

    labelResSem = Label(w, text="vos predictions pour cette semaine sont enregistrees dans vos documents un fichier excel")
    labelResSem.grid(row=6,column=2)

    book = Workbook() #enregistrement aussi dans un fichier excel
    feuil1 = book.add_sheet('feuille 1')
    for i in range(0,len(predicSemaine)-1):
    	feuil1.write(i,0,predicSemaine[i])
    book.save('predictionsParSemaine.xls')

bt2 = Button(w, text="Entrer Semaine", command=affichagePrevisionSemaine)
bt2.grid(row=5, column=2)



#PREDICTIONS PAR SEMAINES PAR ARTICLES

def predictionSemArt():
	semainesVar = []
	indice = lbx.curselection()[0]
	indice2 = lbx2.curselection()[0]
	semainesVar = pd.Series.tolist(resVar[int(indice)])
	labelSemArt = Label(w, text=semainesVar[int(indice2)])
	labelSemArt.grid(row=6,column=3)

bt3 = Button(w, text="Entrer Semaine et Article", command=predictionSemArt)
bt3.grid(row=5, column=3)


##################CREATION DU FICHIER EXCEL #########################################

#if clic sur le bouton download : 

def telechargeData():
	book = Workbook()
	feuil1 = book.add_sheet('feuille 1')
	llPredic = []
	for i in range(0,len(resVar)-1):
		llPredic.append(pd.Series.tolist(resVar[i]))
		for j in range(0,len(llPredic[i])-1):
			feuil1.write(j,i,llPredic[i][j])
	book.save('predictions.xls')

	labelConfirm = Label(w, text="Enregistré !")
	labelConfirm.grid(row=6,column=4)

bt4 = Button(w, text="Télécharger les données", command=telechargeData)
bt4.grid(row=5, column=4)


w.mainloop()
