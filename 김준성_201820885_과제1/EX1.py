from math import cos,pi 
  
positive_result = ["positive var"]
negative_result = ["negative var"]
for i in range(0,36):
  tmp = cos(i*(2*pi)/36) 
  if   tmp > 0:
    positive_result.append(tmp)
  elif tmp < 0:
    negative_result.append(tmp)
  else:
    continue

for i in range(0, len(positive_result)):
  print(positive_result[i], end = '\n')
for j in range(0, len(negative_result)):
  print(negative_result[j], end = "\n")

