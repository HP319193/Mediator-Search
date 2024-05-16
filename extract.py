import csv, json
metadata_list = ['fullname', 'mediator profile on mediate.com', 'mediator Biography', 'mediator state']

def extract_practice():
    csvfile = "updated.csv"

    header_to_extract = "mediator areas of practice"

    values = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if header_to_extract in row:
                text = row[header_to_extract]
                practice_list = text.split('|')

                for practice in practice_list:
                    new_practice = practice.strip()

                    if not new_practice in values and not new_practice.isdigit():
                        values.append(new_practice)

    # jsonfile_path = "practice.json"

    # with open(jsonfile_path, 'w') as file:
    #     json.dump(values, file, indent=4)

    return values

def extract_state():
    csvfile = "updated.csv"

    header_to_extract = "mediator state"

    values = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if header_to_extract in row:
                text = row[header_to_extract]
                
                if not text in values:
                    values.append(text)

    return values

def extract_city():
    csvfile = "updated.csv"

    header_to_extract = "mediator city"
    header_state = "mediator state"
    values = {}
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if header_to_extract in row:
                text = row[header_to_extract]
                if not text in values:
                    values[text] = row[header_state]

    return values 

def search_mediator(filter: dict, practice: str):
    print("filter =>", filter)
    csvfile = "updated.csv"
    mediator_data = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            isMatch = True
            for key, value in filter.items():
                if row[key] != value:
                    isMatch = False
            
            if not practice in row['mediator areas of practice']:
                isMatch = False

            if isMatch:
                data = {}
                for medadata in metadata_list:
                    data[medadata] = row[medadata]

                mediator_data.append(data)
                
    return mediator_data