

#import libraries
import bs4
from bs4 import BeautifulSoup
import urllib.request as urllib2
import pandas as pd

### handle the data
filepath = "C:\\Users\\Abhijit\\Documents\\Abhijit Work\\Draft Value\\New Draft Predictions.txt"
def handlePicksPlayers(filepath):
    picks = pd.read_table(filepath)
    picks = picks[["Player"]]
    picks["Player"] = picks["Player"].apply(lambda x: x.lower())
    picks["Player"] = picks["Player"].apply(lambda x: x.replace(".",""))
    picks["Player"] = picks["Player"].apply(lambda x: x.replace(" ","-"))
    picks["Player"] = picks["Player"].apply(lambda x: x.replace("'",""))
    picks["Player"] = picks["Player"].apply(lambda x:x + "-2.html" if x=="thomas-robinson" else x + "-1.html")
    
    return picks

##### generate the URL
def generateURLs(picks):
    original = "http://www.sports-reference.com/cbb/players/"
    picks["URLs"] = original + picks["Player"]
    return picks

data = handlePicksPlayers(filepath)
newdata = generateURLs(data)

def scrapeData(picks):
    data = pd.DataFrame(columns = ["Player","pts_per_g","fg3_pct","fg2_pct","tov_per_g","blk_per_g","stl_per_g","ast_per_g","trb_per_g","lb","height","years"])
    identities = ["pts_per_g","fg3_pct","fg2_pct","tov_per_g","blk_per_g","stl_per_g","ast_per_g","trb_per_g"]
    for url in picks["URLs"]:
        print(url)
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page,"html.parser")
        storedata = {}
        years = len(soup.find("tbody").find_all("tr"))
        storedata["years"] = [years]
        name = soup.find("h1").text.strip()
        storedata["Player"] = [name]
        prepounds = soup.find_all("span",attrs={"itemprop":"weight"})
        if not(not(prepounds)):
            pounds = float(prepounds[0].text.strip().split("lb")[0])
        else:
            pounds = 0
        storedata["lb"] = [pounds]
        preheight = soup.find_all("span",attrs={"itemprop":"height"})[0].text.strip().split("-")
        height = 12*float(preheight[0]) + float(preheight[-1])
        storedata["height"] = [height]
        for identity in identities:
            pre_name_box = soup.find_all("td",attrs={"data-stat":identity})[-1]
            value = pre_name_box.text.strip()
            if value == "":
                name_box = 0
            else:
                name_box = float(pre_name_box.text.strip())
            storedata[identity] = [name_box]
        df = pd.DataFrame(storedata)
        data = pd.concat([data,df])
    return data


lol = scrapeData(newdata)
def writeToCSV(scrapedData):
    scrapedData.to_csv("Pick Draft Data (Predictions).csv",header = scrapedData.columns.values)
    return None

writeToCSV(lol)

        
            
            
            
            
        
        
