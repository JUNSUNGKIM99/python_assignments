student_info = []
student_size = 10
for i in range(0, student_size):
  student_info.append(input("Name, Student ID, Score:").split(', '))
  student_info[i][1] = int(student_info[i][1])
  student_info[i][2] = int(student_info[i][2])

for i in range(0, student_size):
  print(student_info[i], end = '\n')