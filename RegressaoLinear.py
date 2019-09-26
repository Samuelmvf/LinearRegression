#biblioteca para manipulação de arquivo
import pandas as pd
#biblioteca para realização de operações matemáticas
import numpy as np
#bibliotecas para graficos
import seaborn as sns
import matplotlib.pyplot as plt
#Bibliotecas para calculo da regressão linear
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



#Ler CSV e organizar em tabela
df = pd.read_csv('AnaliseEstudo.csv', sep=';')
df.head()

#Cria coluna Media realizando media através do metado mean. Axis 1 representa o eixo X.
df['media'] = df[['Prova1','Prova2','Prova3']].mean(axis=1)
#Plota o grafico df com exceção da coluna media
#sns.pairplot(df[['TempoEstudo','Faltas','Prova1','Prova2','Prova3','media']])
    
def mediaIdade():
    #Definindo Eixo Y como Média
    y = df['media']
    #Definindo Eixo X como idade
    X = df[['Idade']]
    #Seperando percentual dos dados para treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)#random_state é para a população separada ser sempre a semelhante após escolhida randomicamente
    #Instanciando Regressão Linear
    lr = LinearRegression()
    #Realizando o treino
    lr.fit(X_train, y_train)
    #Faz predições usando a população de teste
    y_pred = lr.predict(X_test)    
    #Coeficientes:
    print('Coeficientes: \nA:', lr.coef_,'B: 0')
    #Desvio padrão
    print("Desvio padrão quadrático: %.2f"
          % mean_squared_error(y_test, y_pred))
    #Taxa de variancia, quanto mais proximo de 1, mais perfeita é a predição
    print('Taxa de variancia: %.2f' % r2_score(y_test, y_pred))   
    #Plotando na tela o gráfico
    
    #População Treinada de Amarelo
    plt.scatter(X_train, y_train,  color='red')
    #População de teste de Preto
    plt.scatter(X_test, y_test,  color='black')
    #Reta de Azul
    plt.plot(X_test, y_pred, color='blue', linewidth=3)    
    
    #plt.xticks(()) #Remove legendas do Eixo X
    #plt.yticks(()) #Remove legendas do Eixo Y  
    
    plt.show()

def mediaTempoEstudo():
    #Definindo Eixo Y como Média
    y = df['media']
    #Definindo Eixo X como Tempo de Estudo
    X = df[['TempoEstudo']]
    #Seperando percentual dos dados para treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Instanciando Regressão Linear
    lr = LinearRegression()
    #Realizando o treino
    lr.fit(X_train, y_train)
    #Faz predições usando a população de teste
    y_pred = lr.predict(X_test)    
    #Coeficientes:
    print('Coeficientes: \n A:', lr.coef_,'B: 0')
    #Desvio padrão
    print("Desvio padrão quadrático: %.2f"
          % mean_squared_error(y_test, y_pred))
    #Taxa de variancia, quanto mais proximo de 1, mais perfeita é a predição
    print('Taxa de variancia: %.2f' % r2_score(y_test, y_pred))    
    #Plotando na tela o gráfico
    #População Treinada de Amarelo
    plt.scatter(X_train, y_train,  color='red')
    #População de teste de Preto
    plt.scatter(X_test, y_test,  color='black')
    #Reta de Azul
    plt.plot(X_test, y_pred, color='blue', linewidth=3)    
    
    #plt.xticks(()) #Remove legendas do Eixo X
    #plt.yticks(()) #Remove legendas do Eixo Y  
    plt.show()
    
def mediaFaltas():
    #Definindo Eixo Y como Média
    y = df['media']
    #Definindo Eixo X como Faltas
    X = df[['Faltas']]
    #Seperando percentual dos dados para treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Instanciando Regressão Linear
    lr = LinearRegression()
    #Realizando o treino
    lr.fit(X_train, y_train)
    #Faz predições usando a população de teste
    y_pred = lr.predict(X_test)   
    #Coeficientes:
    print('Coeficientes: \n A:', lr.coef_,'B: 0')
    #Desvio padrão
    print("Desvio padrão quadrático: %.2f"
          % mean_squared_error(y_test, y_pred))
    #Taxa de variancia, quanto mais proximo de 1, mais perfeita é a predição
    print('Taxa de variancia: %.2f' % r2_score(y_test, y_pred))   
    #Plotando na tela o gráfico
    #População Treinada de Amarelo
    plt.scatter(X_train, y_train,  color='red')
    #População de teste de Preto
    plt.scatter(X_test, y_test,  color='black')
    #Reta de Azul
    plt.plot(X_test, y_pred, color='blue', linewidth=3)  
    #plt.xticks(()) #Remove legendas do Eixo X
    #plt.yticks(()) #Remove legendas do Eixo Y    
    plt.show()
    
print('Media X Idade:\n')
mediaIdade()
print('Media X Tempo de estudo\n')
mediaTempoEstudo()
print('Media X Faltas\n')
mediaFaltas()
