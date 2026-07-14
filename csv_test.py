text = open("export.csv","r").read()

for line in text.splitlines()[1:]:
    time = line.split(",")[0]
    value = line.split(",")[1]
    if float(time) < 1589420000:
        print(time)