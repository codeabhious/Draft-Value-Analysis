# -*- coding: utf-8 -*-
#import libraries
from bs4 import BeautifulSoup
import urllib.request as urllib2
import pandas as pd

### create URLs and DataFrame
originalurl = "http://www.basketball-reference.com/players/"
textfile = "C:\\Users\\Abhijit\\Documents\\Abhijit Work\\Draft Value\\Players and Picks (Validation).txt"
def createData(url,textfile):
    textdata = pd.read_table(textfile)
    textdata = textdata[["Player"]]
    textdata["Player"] = textdata["Player"].apply(lambda x: x.lower())
    textdata["Player"] = textdata["Player"].apply(lambda x: x.replace("'",""))
    textdata["Player"] = textdata["Player"].apply(lambda x: x.replace(".",""))
    textdata["Player"] = textdata["Player"].apply(lambda x: x.split())
    textdata["Player"] = textdata["Player"].apply(lambda x: x[1][0] + "/" + x[1][0:5] + x[0][0:2] + "01.html")
    textdata["URLs"] =  originalurl + textdata["Player"] 
    return textdata

data = createData(originalurl,textfile)

def scrapeWS(picks):
    data = pd.DataFrame(columns = ["Player","WS","Games"])
    #url  = 'http://www.basketball-reference.com/players/c/curryja01.html'
    for url in picks["URLs"]:
        storedata = {}
        print(url)
        try:
            
            page = urllib2.urlopen(url)
            soup = BeautifulSoup(page,"html.parser")
            
            storedata["Name"] = [soup.find("h1").text.strip()]
            prePreGames = soup.find(id="info").text.strip().replace("\n"," ").split("Career    G")
            if len(prePreGames) < 2:
                storedata["Games"] = [0]
            else:
                preGames = prePreGames[1].split(" ")
                for item in preGames:
                    if item =="":
                        preGames.remove(item)
                if ("FG%"in preGames[2]) or ("PTS" in preGames[2]):
                    storedata["Games"] = [float(preGames[1])]
                else:
                    storedata["Games"] = [float(preGames[0])]
            prePreWS = soup.find(id="info").text.strip().replace("\n"," ").split("WS")
            if len(prePreWS) < 2:
                storedata["WS"] = [0]
            else:
                preWS = prePreWS[1].split(" ")
                for item in preWS:
                    if item =="":
                        preWS.remove(item)
                if ("FG%"in preWS[2]) or ("PTS" in preWS[2]):
                    storedata["WS"] = [float(preWS[1])]
                else:
                    storedata["WS"] = [float(preWS[0])]
        except urllib2.URLError:
            storedata["Name"] = [url]
            storedata["Games"] = [0]
            storedata["WS"] = [0]
        print(storedata)
        df = pd.DataFrame(storedata)
        data = pd.concat([data,df])
    return data
lol = scrapeWS(data)

def writeToCSV(scrapedData):
    scrapedData.to_csv("Pick Draft Data WS (2).csv",header = scrapedData.columns.values)
    return None 

writeToCSV(lol)