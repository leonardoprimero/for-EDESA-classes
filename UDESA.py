import numpy as np
from scipy.optimize import minimize
import pandas_datareader.data as web
import pandas as pd


inicio= "2021-02-23"
fin="2021-03-12"

# datosDiariosAAPL =web.get_data_yahoo("AAPL",inicio,fin,interval="d")
# datosDiariosTSLA =web.get_data_yahoo("TSLA",inicio,fin,interval="d")
# datosDiariosMELI =web.get_data_yahoo("MELI",inicio,fin,interval="d")
# datosDiariosFDX =web.get_data_yahoo("FDX",inicio,fin,interval="d")
# datosDiariosMSFT =web.get_data_yahoo("MSFT",inicio,fin,interval="d")
# datosDiariosINTC =web.get_data_yahoo("INTC",inicio,fin,interval="d")
# datosDiariosGOOGL =web.get_data_yahoo("GOOGL",inicio,fin,interval="d")
# datosDiariosAMZN =web.get_data_yahoo("AMZN",inicio,fin,interval="d")
# datosDiariosJNJ =web.get_data_yahoo("JNJ",inicio,fin,interval="d")
# datosDiariosPFE =web.get_data_yahoo("PFE",inicio,fin,interval="d")
# datosDiariosWMT =web.get_data_yahoo("WMT",inicio,fin,interval="d")
# datosDiariosPG =web.get_data_yahoo("PG",inicio,fin,interval="d")
# datosDiariosKO =web.get_data_yahoo("KO",inicio,fin,interval="d")
# datosDiariosC=web.get_data_yahoo("C",inicio,fin,interval="d")
# datosDiariosJPM =web.get_data_yahoo("JPM",inicio,fin,interval="d")
# datosDiariosGE =web.get_data_yahoo("GE",inicio,fin,interval="d")
# datosDiariosXOM =web.get_data_yahoo("XOM",inicio,fin,interval="d")
# datosDiariosX =web.get_data_yahoo("X",inicio,fin,interval="d")

datosDiariosAAPL =web.get_data_yahoo("AAPL.BA",inicio,fin,interval="d")
datosDiariosTSLA =web.get_data_yahoo("TSLA.BA",inicio,fin,interval="d")
datosDiariosMELI =web.get_data_yahoo("MELI.BA",inicio,fin,interval="d")
datosDiariosFDX =web.get_data_yahoo("FDX.BA",inicio,fin,interval="d")
datosDiariosMSFT =web.get_data_yahoo("MSFT.BA",inicio,fin,interval="d")
datosDiariosINTC =web.get_data_yahoo("INTC.BA",inicio,fin,interval="d")
datosDiariosGOOGL =web.get_data_yahoo("GOOGL.BA",inicio,fin,interval="d")
datosDiariosAMZN =web.get_data_yahoo("AMZN.BA",inicio,fin,interval="d")
datosDiariosJNJ =web.get_data_yahoo("JNJ.BA",inicio,fin,interval="d")
datosDiariosPFE =web.get_data_yahoo("PFE.BA",inicio,fin,interval="d")
datosDiariosWMT =web.get_data_yahoo("WMT.BA",inicio,fin,interval="d")
datosDiariosPG =web.get_data_yahoo("PG.BA",inicio,fin,interval="d")
datosDiariosKO =web.get_data_yahoo("KO.BA",inicio,fin,interval="d")
datosDiariosC=web.get_data_yahoo("C.BA",inicio,fin,interval="d")
datosDiariosJPM =web.get_data_yahoo("JPM.BA",inicio,fin,interval="d")
datosDiariosGE =web.get_data_yahoo("GE.BA",inicio,fin,interval="d")
datosDiariosXOM =web.get_data_yahoo("XOM.BA",inicio,fin,interval="d")
datosDiariosX =web.get_data_yahoo("X.BA",inicio,fin,interval="d")


def Rendimientos(ListaPrecios):
     rendimiento = []
     for j in range(1, len(ListaPrecios)):
         rendimiento.append(ListaPrecios[j]/ListaPrecios[j-1]-1)
     return rendimiento
 
RendAAPL = Rendimientos(datosDiariosAAPL["Adj Close"])
RendTSLA = Rendimientos(datosDiariosTSLA["Adj Close"])
RendMELI = Rendimientos(datosDiariosMELI["Adj Close"])
RendFDX = Rendimientos(datosDiariosFDX["Adj Close"])
RendMSFT = Rendimientos(datosDiariosMSFT["Adj Close"])
RendINTC = Rendimientos(datosDiariosINTC["Adj Close"])
RendGOOGL = Rendimientos(datosDiariosGOOGL["Adj Close"])
RendAMZN = Rendimientos(datosDiariosAMZN["Adj Close"])
RendJNJ = Rendimientos(datosDiariosJNJ["Adj Close"])
RendPFE = Rendimientos(datosDiariosPFE["Adj Close"])
RendWMT = Rendimientos(datosDiariosWMT["Adj Close"])
RendPG = Rendimientos(datosDiariosPG["Adj Close"])
RendKO = Rendimientos(datosDiariosKO["Adj Close"])
RendC = Rendimientos(datosDiariosC["Adj Close"])
RendJPM = Rendimientos(datosDiariosJPM["Adj Close"])
RendGE = Rendimientos(datosDiariosGE["Adj Close"])
RendXOM = Rendimientos(datosDiariosXOM["Adj Close"])
RendX = Rendimientos(datosDiariosX["Adj Close"])
def FuncionEjemplo (rendimiento):
    Promedio=sum(rendimiento)/len(rendimiento)
    return Promedio


PromedioAAPL = FuncionEjemplo(RendAAPL)
PromedioTSLA = FuncionEjemplo(RendTSLA)
PromedioMELI = FuncionEjemplo(RendMELI)
PromedioFDX = FuncionEjemplo(RendFDX)
PromedioMSFT = FuncionEjemplo(RendMSFT)
PromedioINTC = FuncionEjemplo(RendINTC)
PromedioGOOGL = FuncionEjemplo(RendGOOGL)
PromedioAMZN = FuncionEjemplo(RendAMZN)
PromedioJNJ = FuncionEjemplo(RendJNJ)
PromedioPFE = FuncionEjemplo(RendPFE)
PromedioWMT = FuncionEjemplo(RendWMT)
PromedioPG = FuncionEjemplo(RendPG)
PromedioKO = FuncionEjemplo(RendKO)
PromedioC = FuncionEjemplo(RendC)
PromedioJPM = FuncionEjemplo(RendJPM)
PromedioGE = FuncionEjemplo(RendGE)
PromedioXOM = FuncionEjemplo(RendXOM)
PromedioX = FuncionEjemplo(RendX)

print ("PromedioAAPL",PromedioAAPL)
print ("PromedioTSLA",PromedioTSLA)
print ("PromedioMELI",PromedioMELI)
print ("PromedioFDX",PromedioFDX)
print ("PromedioMSFT",PromedioMSFT)
print ("PromedioINTC",PromedioINTC)
print ("PromedioGOOGL",PromedioGOOGL)
print ("PromedioAMZN",PromedioAMZN)
print ("PromedioJNJ",PromedioJNJ)
print ("PromedioPFE",PromedioPFE)
print ("PromedioWMT",PromedioWMT)
print ("PromedioPG",PromedioPG)
print ("PromedioKO",PromedioKO)
print ("PromedioC",PromedioC)
print ("PromedioJPM",PromedioJPM)
print ("PromedioGE",PromedioGE)
print ("PromedioXOM",PromedioXOM)
print ("PromedioX",PromedioX)


def objetive (w)  :
    r=  (0.08,0.05,0.07)     #( PromedioAAPL, PromedioEPOR, PromedioWFC )
    r= np.array(r)
    return -(r @ w)

def restriccion1(w):
    sum_sq=1
    for i in range (3):
        sum_sq=sum_sq-w[i]
    return sum_sq
rest1 = {"type":"eq","fun":restriccion1}

# Matríz de Retornos Diarios
MatrizRendimientos = [ RendAAPL, RendEPOR, RendWFC ]
C=np.cov(MatrizRendimientos)
C=np.matrix(C)
def restriccion2 (w) :
    w = np.array(w)[np.newaxis]
    Sigma_P = np.sqrt(w*C*w.T*252)
    Sigma_objetivo = 0.22
    S = Sigma_P - Sigma_objetivo
    return float (S)

rest2 = {"type" : "eq" , "fun" : restriccion2}

def restriccion3 (w) :
    return w[0]+w[1]-0.2

rest3 = {"type" : "ineq", "fun" : restriccion3}

restricciones = [rest1, rest2, rest3]

# Minimos y Maximos de los Pesos
b1 = (0.0,0.7)
b2 = (0.0,0.5)
b3 = (0.2,0.4)
bnds =(b1,b2,b3)

w0 = [0.0,0.0,0.0]

sol = minimize(objetive,w0,method= "SLSQP", bounds=bnds,constraints=restricciones)
Retorno_Port_opt = -1*sol.fun
Weigth_Port_opt = sol.x

#muestro la solucion 
print (sol)
print ("Retorno_Port_opt=" , Retorno_Port_opt)
print ("Weigth_Port_opt=",Weigth_Port_opt)

my_array = np.array([Weigth_Port_opt])
df = pd.DataFrame(my_array, columns = ['AAPL','EPOR','WFC'])

print(df)
print(type(df))                 
#df_excel = df.to_excel("pesoscon3df.xlsx")   