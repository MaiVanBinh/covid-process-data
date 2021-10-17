f = open("./data/test_split_3.txt", "r")
for x in f:
  imageName =  x.split()[1] 
  label = x.split()[2] 
  print(imageName)
  print(label)