import os
from pathlib import Path

folder = {"workshop_gallery_satellite": "./data/test",
        "workshop_query_street": "./data/test"}
txt_file = {"workshop_gallery_satellite": "gallery_name.txt",
        "workshop_query_street": "query_street_name.txt"}
        
for key, value in folder.items():
    with open(txt_file[key], 'w') as f:
        for root,_,files in os.walk(value):
            for file in files:
                file_path = os.path.join(root, file)
                # Convert to Unix-style slashes
                file_path = file_path.replace(os.sep, '/')
                f.write(f"{file_path}\n")
                print(file_path)
