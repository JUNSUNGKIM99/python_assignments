str_origin = "C:\\USER\\.HATECOVID19\\KONO\\GAGOSIPDA\\python\\example.py".split('\\')
str_1 = str_origin[:]
str_2 = str_origin[:]
str_3 = str_origin[:]

str_1[-1] = "example_test.py"
str_1 = '\\'.join(str_1)
str_2[6:] = "new_folder example.py".split()
str_2 = '\\'.join(str_2)
str_3[-1] = "new_name.py"
str_3 = '\\'.join(str_3)

print(str_1)
print(str_2)
print(str_3)