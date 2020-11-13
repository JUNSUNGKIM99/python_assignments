list1 = [1, 2, 3, [4, 5, 6]]
list2 = list1
print("#1")
print("l1", list1)
print("l2", list2)
list2[2] = 10
print("#2")
print("l1", list1)
print("l2", list2)
list1_copy = list1[:]
list1_copy[3][0] = 99
print("#3")

print("l1", list1)
print("l2", list2)
print("l3", list1_copy)