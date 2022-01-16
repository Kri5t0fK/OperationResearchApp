import random
import datetime
import json

quantities = [0, 100, 200, 300] # quantities describe discrete product amounts


def generate_prices(m: int):
    l = []
    for i in range(m):
        l.append(random.randint(5, 29))
    return f'"prices": {json.dumps(l)},'


def generate_quantities(m: int):
    l = []
    for i in range(m):
        l.append(quantities[random.randint(0, 3)])
    return f'"quantities": {json.dumps(l)},'


def generate_exp_dates(m: int):
    l = []
    for i in range(m):
        l.append(random.randint(1, 6))
    return f'"dates": {json.dumps(l)},'


def generate_recipes(n: int, m: int):
    l = []
    for i in range(n):
        nl = []
        for j in range(m):
            aux = random.randint(-50, 10)
            aux = 0 if aux<0 else aux
            nl.append(aux)
        l.append(nl)
    return f'"recipes": {json.dumps(l)},'


def generate_times(n: int):
    l = []
    for i in range(n):
        l.append(random.randint(5, 31))
    return f'"times": {json.dumps(l)}'


def save_file(json_obj):
	date_for_filename = str(datetime.datetime.utcnow())
	date_for_filename = date_for_filename.split(' ')[1]
	date_for_filename = date_for_filename.split('.')[0]
	date_for_filename = date_for_filename.replace(':', '_')

	with open(f'generated_data/{date_for_filename}.json', 'w') as file:
		file.write(json_obj)


def generate_model():
	random.seed()
	n = 500000
	m = 100
	money = 1000

	json_object = '{' \
                  + f'"n": {n},' \
                  + f'"m": {m},' \
                  + f'"money": {money},' \
                  + generate_prices(m) \
                  + generate_quantities(m) \
                  + generate_exp_dates(m) \
                  + generate_recipes(n, m) \
                  + generate_times(n) \
                  + '}'
	
	save_file(json_object)

if __name__ == "__main__":
    generate_model()
	