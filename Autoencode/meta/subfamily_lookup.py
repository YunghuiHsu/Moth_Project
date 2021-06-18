import urllib.request
import re
import time

prog1 = re.compile('(GenusDetails.dsml\?.+?)&')
prog2 = re.compile('\s+(.+?)\<\/changecase\>')
genus_higher_taxa = dict()

genus_higher_taxa

for genus in genus_list:
    q1 = f"https://www.nhm.ac.uk/our-science/data/butmoth/search/GenusList3.dsml?GENUSqtype=equals&GENUS={genus}"
    try:
        higher_taxa = genus_higher_taxa[genus]
        print("Ignoring", genus)
    except:
        print("Fetching", genus)
        failed_counter = 0
        while True:
            try:
                with urllib.request.urlopen(q1) as response1:
                    html1 = response1.read()
                    s1 = prog1.search(html1.decode('utf-8'))
                    if s1 is None:
                        genus_higher_taxa[genus] = "Unknown"
                    else:
                        q2 = "https://www.nhm.ac.uk/our-science/data/butmoth/search/" + s1.group(1)
                        with urllib.request.urlopen(q2) as response2:
                            html2 = response2.read()
                            s2 = prog2.search(html2.decode('utf-8'))
                            higher_taxa = s2.group(1)
                            genus_higher_taxa[genus] = higher_taxa.replace(' ', '')
                break
            except:
                failed_counter += 1
                time.sleep(10)
                if failed_counter == 5:
                    break

genus_higher_taxa.keys()

pd.DataFrame({'genus': list(genus_higher_taxa.keys()), 'higher': list(genus_higher_taxa.values())}).to_csv('genus_to_higher.csv', index=False, sep='\t')