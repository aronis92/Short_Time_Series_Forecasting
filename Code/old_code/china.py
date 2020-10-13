# import netcdf4 as nc

data = []
with open("data/China_Station_SO2.txt", "r") as f:
    data = f.readlines()
f.close()
del f

headline = data.pop(0)


for i in range(len(data)):
    data[i] = data[i].split()
    #data[i] = data[i][4:]
    
sample = data[1000:1500]