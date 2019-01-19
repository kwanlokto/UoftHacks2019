import csv


def readCSV(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        data = []
        for row in csv_reader:
            if line_count > 0:
                numbers = [int(x) for x in row]
                data.append(numbers)
            line_count += 1
        print(f'Processed {line_count} lines.')
        # print(data)
        return data
