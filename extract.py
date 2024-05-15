import csv, json

def extract_practice():
    csvfile = "test.csv"

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

    jsonfile_path = "practice.json"

    with open(jsonfile_path, 'w') as file:
        json.dump(values, file, indent=4)

    return values

def extract_state():
    csvfile = "test.csv"

    header_to_extract = "mediator state"

    values = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if header_to_extract in row:
                text = row[header_to_extract]
                
                if not text in values:
                    values.append(text)

    jsonfile_path = "state.json"

    with open(jsonfile_path, 'w') as file:
        json.dump(values, file, indent=4)

    return values

def extract_city():
    csvfile = "test.csv"

    header_to_extract = "mediator city"

    values = []
    with open(csvfile, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if header_to_extract in row:
                text = row[header_to_extract]
                
                if not text in values:
                    values.append(text)

    jsonfile_path = "city.json"

    with open(jsonfile_path, 'w') as file:
        json.dump(values, file, indent=4)

    return values 