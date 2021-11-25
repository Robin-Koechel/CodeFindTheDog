#https://www.youtube.com/watch?v=s4jtkzHhLzY

from bs4 import BeautifulSoup as bs
import requests
URL = "https://www.tasso.net/Tierregister/Suchmeldungen"
page = requests.get(URL)
html = page.content

soup = bs(html,'html.parser')
suchmeldungen = soup.find_all("div", {"class": "columns medium-6 suchmeldung-list"})
soup = bs(str(suchmeldungen),'html.parser')
meldung = soup.findAll('article', {"class": "row"})


#print(meldung[1])
for i in range(len(meldung)):
    soup = bs(str(meldung[i]),'html.parser')
    info = soup.findAll('span')
    #info.pop(3)
    petBreed = str(info[i]).strip("<span>").strip("</span>")
    print(petBreed)