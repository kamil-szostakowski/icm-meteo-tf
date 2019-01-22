import os
import urllib2
import datetime

# http://www.meteo.pl/um/metco/mgram_pict.php?ntype=0u&fdate=2018051812&row=383&col=209&lang=pl

""" Directory when we will store all the downloaded meteorograms """
destinamtion_dir = 'data/prediction-images/'

""" We skip files which are already present in this directory """
filter_dir = 'data/source-images/'

""" Maximum number of days in the past we can reach for """
distant_past = 156

""" All the main cities we use as data sources """
locations = [
    [379, 285], # Bialystok
    [381, 199], # Bydgoszcz
    [346, 210], # Gdansk
    [390, 152], # Gorzow Wielkopolski
    [461, 215], # Katowice
    [443, 244], # Kielce
    [466, 232], # Krakow
    [418, 223], # Lodz
    [432, 277], # Lublin
    [363, 240], # Olsztyn
    [449, 196], # Opole
    [400, 180], # Poznan
    [465, 269], # Rzeszow
    [370, 142], # Szczecin
    [383, 209], # Torun
    [406, 250], # Warszawa
    [436, 181], # Wroclaw
    [412, 155], # Zielona Gora
]

def get_url(location, date, time):
    return 'http://www.meteo.pl/um/metco/mgram_pict.php?ntype=0u&fdate={date!s}{time:02d}&row={row!s}&col={col!s}&lang=pl'.format(
            date=date.strftime("%Y%m%d"),
            time=time,
            row=location[0],
            col=location[1]
        ) , '{date!s}{time:02d}-{row!s}-{col!s}.png'.format(
            date=date.strftime("%Y%m%d"),
            time=time,
            row=location[0],
            col=location[1]            
        )

def download_meteorogram(url, destination_path):
    print('Downloading {url} => {file}'.format(url=url, file=destination_path))
    try:
        response = urllib2.urlopen(url)
        with open(destination_path, 'wb') as file_writer:
            file_writer.write(response.read())
    except urllib2.HTTPError, error:
        print "HTTP Error:", error.code, url
    except urllib2.URLError, error:
        print "URL Error:", error.reason, url

# Execution section
if __name__ == "__main__":
    if not os.path.exists(destinamtion_dir):
        os.makedirs(destinamtion_dir)

    for location in locations:
        for day in range(0, distant_past):
            date = datetime.date.today()-datetime.timedelta(days=day)
            for time in range(0, 24, 6):
                url , filename = get_url(location, date, time)
                download_path = os.path.join(destinamtion_dir, filename)
                filter_path = os.path.join(filter_dir, filename)

                if not os.path.exists(filter_path) and not os.path.exists(download_path):
                    download_meteorogram(url,download_path)
                else:
                    print('Skipped download of {file}'.format(file=filename))
