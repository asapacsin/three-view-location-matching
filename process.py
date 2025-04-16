import os
from pathlib import Path

folder = {"workshop_gallery_satellite": "./data/test/workshop_gallery_satellite",
        "workshop_query_street": "./data/test/workshop_query_street"}
txt_file = {"workshop_gallery_satellite": "gallery_name.txt",
        "workshop_query_street": "query_street_name.txt"}
        
for key, value in folder.items():
    with open(txt_file[key], 'w') as f:
        for root,_,files in os.walk(value):
            for file in files:
                f.write(f"{file}\n")
                print(file)
