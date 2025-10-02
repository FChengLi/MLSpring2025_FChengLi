import requests

def dl(url, out):
    r = requests.get(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    with open(out, 'wb') as f:
        f.write(r.content)

dl("https://www.dropbox.com/s/lmy1riadzoy0ahw/covid.train.csv?dl=1", "covid_train.csv")
dl("https://www.dropbox.com/s/zalbw42lu4nmhr2/covid.test.csv?dl=1",  "covid_test.csv")
