201820885 김준성 프로그래밍과제2 제출

직접 작성한 코드를 제출하였으며 

예외처리를 한 것을 확인하기위해 조금씩 example.txt를 바꿔가며 실행한 터미널의 모습입니다. 

그리고 마지막 2개의 결과중 전자는 example프로그램이 정상적으로 작동한 것을 보인 것이고 
마지막 결과는 instruction을 2개 추가한 결과입니다. 
추가한 결과는 
mul $4, $2, $1
st  %4, 6

을 추가하여 실행한 결과입니다.

성공적으로 출력된 것을 확인할 수 있습니다.

예외처리를 구현하는 것에 있어서는 
산술처리 || 메모리 읽고,쓰기 || 분기점 
이 3가지 묶음으로 예외처리를 구현했습니다. 
즉, 산술의 add sub mul div의 예외처리는 같은 코드를 가지고 구현했고
메모리 읽고,쓰기는 ld st의 예외처리는 같은 코드를 가지고
분기점 jump beq ble Ben 등의 기능은 비슷한 예외처리 방식으로 구현하였습니다.

오류가 발생하거나, 다른학우들의 코드와 비교해보시고, 저의 코드에 대한 피드백이 어떤 것이 있는지 알려주시면 감사하겠습니다. 