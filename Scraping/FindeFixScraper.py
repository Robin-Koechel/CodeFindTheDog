class FindeFix:
    from bs4 import BeautifulSoup as bs
    import requests
    import os
    import Scraping.Pet as Pet
    import sys

    WebsiteUrl = ""
    SuchmeldungsIndex = 0
    lstPets = []

    def __init__(self):
        self.WebsiteUrl = "https://www.findefix.com/"
        self.SuchmeldungsIndex = 1

    def scrapePets(self, zuDurchsuchendeSeiten):
        for SuchmeldungsIndex in range(zuDurchsuchendeSeiten):
            page = self.requests.get("https://www.findefix.com/haustier-vermisst-gefunden/aktuelle-suchmeldungen/" \
                                    "?tx_dhrvermisst_vermisst%5B@widget_0%5D%5BcurrentPage%5D="+str(SuchmeldungsIndex))
            html = page.content
            soup = self.bs(html, 'html.parser')
            suchmeldungen = soup.find_all("div", {"class": "meldung ltrow pad"})
            for meldung in suchmeldungen:
                #FIND Images
                for img in meldung.find_all("img"):
                    filename = img.attrs.get("src")
                    imgUrl = self.WebsiteUrl + filename

                    if "fileadmin/content/vermisstbilder/" in filename:
                        self.downloadImage(imgUrl)#download Images

                        #INFO PETs
                        infoPet = meldung.find_all("div", {"class": "col lg-9 vs-8"})
                        result = []
                        for i in infoPet:
                            cut1 = str(i).replace('<div class="col lg-9 vs-8">', "")
                            cut2 = cut1.replace("</div>","")
                            if "XXXXXXX" not in cut2 and "vorhanden" not in cut2:
                                result.append(cut2)
                        pet = self.Pet.Pet(result[0],result[1],result[3],result[2],result[4],str("FindeFixImages/"+filename[33:]))
                        self.lstPets.append(pet)



    def downloadImage(self,url):
        response = self.requests.get(url)
        name = self.os.path.join("F:\PYTHON\Projekte\FindTheDog\Scraping\FindeFixImages\\", url.split("/")[-1])
        with open(name, "wb") as f_out:
            f_out.write(response.content)

    def getPetLst(self):
        return self.lstPets

#findeFix = FindeFix()
#findeFix.scrapeAll(2)
#lst = findeFix.getPetLst()
#pet1 = lst[0]
#print(pet1.outAll())
