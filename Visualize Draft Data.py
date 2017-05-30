# -*- coding: utf-8 -*-
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



filepath = "C:\\Users\\Abhijit\\Documents\\Abhijit Work\\Draft Value\\Pick Draft Data (Filled 2).csv"

def handle_data(filepath):
    data = pd.read_csv(filepath)
    data["WS/G"] = data["WS"]/data["Games"]
    data["WS/G"] = data["WS/G"].fillna(0)
    #data["Round"] = (data["Pick"] >= 30) + 1
    return data

def visYrsPicks(data):
    sns.plt.title("Years in College vs. Draft Position")
    ax = sns.swarmplot(x = "years",y= "Pick",data=data)
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("Years in College vs. Draft Position.png")
    return None

data = handle_data(filepath)
#print(data["Round"])
#visYrsPicks(data)

def visWSYrs(data):
    sns.plt.title("Years in College vs. Average Win Shares")
    ax = sns.swarmplot(x = "years",y= "WS/G",data=data)
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("Years in College vs. Average Win Shares.png")
    return None

#visWSYrs(data)

def visWSPicks(data):    
    ax = sns.regplot(x = "Pick",y= "WS/G",data=data,fit_reg = False)
    sns.plt.title("Draft Position vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("Draft Position vs. Average Win Shares.png")
    return None


#visWSPicks(data)

def visWSPicksYears(data):
    years = list(range(1,5))
    for year in years:
        yeardata = data.loc[data['years']==year]
        ax = sns.regplot(x = "Pick",y= "WS/G",data=yeardata,fit_reg = True)
        sns.plt.title("Draft Position vs. Average Win Shares: " + str(year) + ' College Years')
        plt.show(ax)
        t = "Draft Position vs. Average Win Share " + "("+str(year) + " College Years).png"
        print(t)
        fig = ax.get_figure()
        fig.savefig(t)
    return None
#visWSPicksYears(data)

def visWSHeight(data):
    ax = sns.swarmplot(x = "height",y = "WS/G",data=data)
    sns.plt.title("College Height vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Height vs. Average Win Shares.png")
    return None

#visWSHeight(data)
def visWSWeight(data):
    ax = sns.regplot(x = "lb",y = "WS/G",data=data)
    sns.plt.title("College Weight vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Weight vs. Average Win Shares.png")
    return None
#visWSWeight(data)

def visWSPPG(data):
    ax = sns.regplot(x = "pts_per_g",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College PPG vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College PPG vs. Average Win Shares.png")
    return None
#visWSPPG(data)
def visWSAPG(data):
    ax = sns.regplot(x = "ast_per_g",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College Assists vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Assists vs. Average Win Shares.png")
    return None
#visWSAPG(data)
def visWSRPG(data):
    ax = sns.regplot(x = "trb_per_g",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College Rebounds vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Rebounds vs. Average Win Shares.png")
    return None
#visWSRPG(data)
def visWSBPG(data):
    ax = sns.regplot(x = "blk_per_g",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College Blocks vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Blocks vs. Average Win Shares.png")
    return None
#visWSBPG(data)
def visWSSPG(data):
    ax = sns.regplot(x = "stl_per_g",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College Steals vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Steals vs. Average Win Shares.png")
    return None
#visWSSPG(data)  
def visWSTOPG(data):
    ax = sns.regplot(x = "tov_per_g",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College Turnovers vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College Turnovers vs. Average Win Shares.png")
    return None
#visWSTOPG(data)
def visWSFG2(data):
    ax = sns.regplot(x = "fg2_pct",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College 2 Point FG Pct. vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College 2 Point FG Pct. vs. Average Win Shares.png")
    return None
#visWSFG2(data)
def visWSFG3(data):
    ax = sns.regplot(x = "fg3_pct",y = "WS/G",data=data,fit_reg = True)
    sns.plt.title("College 3 Point FG Pct. vs. Average Win Shares")
    plt.show(ax)
    fig = ax.get_figure()
    fig.savefig("College 3 Point FG Pct. vs. Average Win Shares.png")
    return None
#visWSFG3(data)

def visPickWeight(data):
    data = data[0:441]
    data["Pick"] = data["Pick"].apply(lambda x:1 if x <=10 else x)
    data["Pick"] = data["Pick"].apply(lambda x:2 if (x <=20 and x >10) else x)
    data["Pick"] = data["Pick"].apply(lambda x:3 if (x <=30 and x >20) else x)
    data["Pick"] = data["Pick"].apply(lambda x:4 if (x <=40 and x >30) else x)
    data["Pick"] = data["Pick"].apply(lambda x:5 if (x <=50 and x >40) else x)
    data["Pick"] = data["Pick"].apply(lambda x:6 if (x <=60 and x >50) else x)
    ax = sns.factorplot(x="height",data=data,kind="count",col="Pick",ci=None,col_wrap=3)
    #sns.plt.suptitle("College Height vs. Pick")
    plt.xticks(rotation=90)
    plt.show(ax)
    #fig.savefig("College Height vs. Pick.png")
    return None
#visPickWeight(data)

def visPickYear(data):
    data = data[0:441]
    data["Pick"] = data["Pick"].apply(lambda x:1 if x <=10 else x)
    data["Pick"] = data["Pick"].apply(lambda x:2 if (x <=20 and x >10) else x)
    data["Pick"] = data["Pick"].apply(lambda x:3 if (x <=30 and x >20) else x)
    data["Pick"] = data["Pick"].apply(lambda x:4 if (x <=40 and x >30) else x)
    data["Pick"] = data["Pick"].apply(lambda x:5 if (x <=50 and x >40) else x)
    data["Pick"] = data["Pick"].apply(lambda x:6 if (x <=60 and x >50) else x)
    ax = sns.factorplot(x="years",data=data,kind="count",col="Pick",ci=None,col_wrap=3)
    #sns.plt.suptitle("College Years vs. Pick")
    plt.xticks(rotation=90)
    plt.show(ax)
    #fig.savefig("College Height vs. Pick.png")
    return None
#visPickYear(data)


def prepareNeuralNet(data):
    #one hot encode the school of attendances
    data_with_dum = pd.get_dummies(data,columns=["School","years"])
    return data_with_dum
data_with_dum = prepareNeuralNet(data)
columns = ["ast_per_g","blk_per_g","fg2_pct","fg3_pct","height","lb","pts_per_g","stl_per_g","tov_per_g","trb_per_g"]
def standardize(data_with_dum,columns):
    import numpy as np
    for col in columns:
        #data_with_dum[col] = (data_with_dum[col] - min(data_with_dum[col]))/(max(data_with_dum[col])-min(data_with_dum[col]))
        data_with_dum[col] = (data_with_dum[col]-np.mean(data_with_dum[col]))/(np.std(data_with_dum[col]))
    return data_with_dum

data_with_dum = standardize(data_with_dum,columns)
def basicNeuralNet(data,batch_size,epochs):
    from keras.models import Sequential
#    import numpy as np
    from keras.layers import Dense
    from sklearn.cross_validation import train_test_split
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="val_loss",patience=5)
    
    datakeep = data[0:441]
    datapred = data[441:]
    
    y = datakeep["WS/G"].as_matrix()
    X = datakeep.drop(labels = ["Player","Pick","Games","WS","WS/G"],axis=1,inplace=False).as_matrix()
    X_1 = datapred.drop(labels = ["Player","Pick","Games","WS","WS/G"],axis=1,inplace=False).as_matrix()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Dense(8,input_dim = 141))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(8,activation="relu"))
    model.add(Dense(1,activation="linear"))
    model.compile(loss="mean_absolute_error",optimizer= "adam")
    model.fit(X_train,y_train,batch_size= batch_size,epochs = epochs, validation_data = (X_test,y_test),callbacks = [es])
    score = model.evaluate(X_test,y_test,verbose=0)
    print("SCORE:")
    print(score)
    print("PREDICTIONS")
    pred = model.predict(X_1)
    print(pred)
#    print(np.linalg.norm(pred-datapred["WS/G"]))
    return model,pred,datapred
    
model,pred,datapred = basicNeuralNet(data_with_dum,30,1000)

def rankDraftValue(pred,datapred):
    datapred["Projected Win Shares"] = pred
    newdata = datapred[["Player","Projected Win Shares"]]
    sortnewdata = newdata.sort(columns = ["Projected Win Shares"],ascending=False)
    return sortnewdata,newdata
sortnewdata,newdata = rankDraftValue(pred,datapred)

def visProjDraft(sortnewdata):
    
    g = sns.barplot(x="Player",y="Projected Win Shares",data=sortnewdata)
    plt.xticks(rotation = 90)
   # plt.title("Projected Win Sharesfor 2017 NBA Draft")
    plt.show(g)
    #fig.savefig("Projected Win Shares for 2017 NBA Draft.png")
    return None

visProjDraft(sortnewdata)
def prepareNeuralNetSoftMax(data_with_dum):
    data_with_dum["Pick"] = data_with_dum["Pick"].apply(lambda x:1 if x <= 30 else 2)
    data_with_dum = pd.get_dummies(data_with_dum,columns=["Pick"])
    data_with_dum = data_with_dum.drop(labels=["WS","Games","WS/G"],axis=1)
    return data_with_dum

def NeuralNetSoftMax(data,epochs,batch_size):
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.cross_validation import train_test_split  
    datakeep = data[0:441]
    datapred = data[441:]
    X_1 = datapred.drop(labels = ["Player","Pick_1","Pick_2"],axis=1,inplace=False).as_matrix()
    y = datakeep[["Pick_1","Pick_2"]].as_matrix()
    X = datakeep.drop(labels = ["Player","Pick_1","Pick_2"],axis=1,inplace=False).as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    model = Sequential()
    model.add(Dense(1,input_dim=141))
    model.add(Dense(100,activation="relu"))
    model.add(Dense(2,activation="softmax"))
    model.compile(loss="binary_crossentropy",optimizer= "adam",metrics=['mae','acc'])
    model.fit(X_train,y_train,batch_size= batch_size,epochs = epochs, validation_data = (X_test,y_test))
    score = model.evaluate(X_test,y_test,verbose=0)
    print("SCORE:")
    print(score)
    print("PREDICTIONS")
    pred = np.round(np.round(model.predict(X_1),2)*100)
    #players = datapred["Player"]
    return model,pred,datapred
#datasoft = prepareNeuralNetSoftMax(data_with_dum)

#model,pred,datapred= NeuralNetSoftMax(datasoft,50,30)

def visClassification(newdata,pred):
    sns.set_style("whitegrid")
    import numpy as np  
    high = []
    for row in pred:
        print(row[0],row[1])
        if abs(row[0]-row[1]) <= 15:
            high.append(1.5)
        else:
            high.append(np.argmax(row)+1)
    df = pd.DataFrame(columns=["Tier"],data=high)
    newdata["Tier"] = df["Tier"].values
    sortnewdata = newdata.sort(columns=["Projected Win Shares"],ascending=False)
    g = sns.barplot(x="Player",y="Projected Win Shares",data=sortnewdata,hue="Tier",palette = "Dark2")
    plt.xticks(rotation = 90)
    #plt.title("Projected Win Shares and Tiers for 2017 NBA Draft")
    plt.show(g)
    #fig.savefig("Projected Win Shares and Tiers for 2017 NBA Draft.png")
    return None
#visClassification(newdata,pred)

    
        