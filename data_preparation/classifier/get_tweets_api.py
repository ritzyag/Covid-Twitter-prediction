import gzip
import json
import os
from twarc import Twarc
from tqdm import tqdm
import warnings
import logging
import schedule
import time
from datetime import datetime, timedelta


############# READ KEYS
with open('../../files/keywords.txt') as fp:
    cities = fp.readline().split()
    keywords = fp.read().split() + ['corona '+ x.lower() for x in cities] + ['covid ' + x.lower() for x in cities]+ ['lockdown ' + x.lower() for x in cities]
    keywords = ['(' + key.strip() + ')' for key in keywords]

############## START

def fetch_lastday(keywords, OUTDIR, outprefix = '', date = None):        
        if date:
                e = datetime.strptime(date, '%Y-%m-%d')
        else:
                e = datetime.now()

        print('START time:', datetime.now())

        s = e - timedelta(days = 1)
        STARTDATE = s.strftime('%Y-%m-%d')
        ENDDATE = e.strftime('%Y-%m-%d')
        
        print('\n'+ STARTDATE)
        
        since = ' since:' + STARTDATE if STARTDATE else ''
        until = ' until:' + ENDDATE if ENDDATE else ''


        
        try: os.mkdir(OUTDIR)
        except: pass

        warnings.filterwarnings("ignore")
        logging.disable()

        i = 0
        idset = set()
        twarc = Twarc(app_auth = True)
        fout = gzip.open(os.path.join(OUTDIR, outprefix + '%s.jsonl.gz'%STARTDATE), 'w')
        for kstart in range(0, len(keywords), 10):
                keys = ' OR '.join(keywords[kstart : kstart + 10])
                print("\n\nkeys:", keys, flush = True)
                for _, tweet in tqdm(enumerate(twarc.search(keys + since + until))):
        
                        if tweet['id'] not in idset:
                                i += 1
                                idset.add(tweet['id'])
                                fout.write(json.dumps(tweet).encode('utf8') + b'\n')
        
                
        if fout: fout.close()
        print()
        e = datetime.now()
        print('END time:', e)
        print('\n\n\n')



def fetch_between(keywords, start, end, OUTDIR):
        s = datetime.strptime(start, '%Y-%m-%d')
        e = datetime.strptime(end, '%Y-%m-%d')

        while s < e:
                s = s + timedelta(days = 1)
                fetch_lastday(keywords, OUTDIR, date = s.strftime('%Y-%m-%d'))


start_date = "date_begining" #change here
end_date = "date_end" #change here

if __name__ == '__main__':
        fetch_between(keywords, start_date, end_date, 'tweets/')


        schedule.clear()
        schedule.every().day.at("08:00").do(fetch_lastday, keywords = keywords, OUTDIR = 'tweets/')


        print('\n\n########## SCHEDULED ############\n')
        while True:
                time.sleep(300)
                schedule.run_pending()
                
