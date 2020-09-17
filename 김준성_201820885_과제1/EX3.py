import copy

a = [1, [6, 5, 2]]
b = a[:]    # shallow copy 실행 
print(b)    # [1, [6, 5, 2]] 출력
b[0] = 100

print(b)    # [100, [1, 2, 3]] 출력,
print(a)    # [1, [6, 5, 2]] 출력, shallow copy 가 발생해 복사된 리스트는 다른 object를 바라보게된다. 그러나 내부요소가 바라보는 리스트는 카피되지않았다. 표먼적인 보이는 것이 복제되었기 때문이다.

c = copy.deepcopy(a) #deep copy 실행한다.  
c[1].append(4)    # 리스트의 두번째 nested_list(내부리스트)에 4를 추가
print(c)    # [1, [6, 5, 2, 4]] 출력
print(a)    # [1, [6, 5, 2]] 출력, a가 c와 똑같이 수정된 이유는 리스트의 item 내부의 객체는 동일한 객체이므로 mutable한 리스트를 수정할때는 둘다 값이 변경됨