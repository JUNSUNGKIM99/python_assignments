#!/usr/bin/python3.8

import sys


class SimpleCPU:
    """
    SimpleCPU 모듈
            레지스터 크기 num_regs 와 mem_size 값을 받는다.
    """

    def __init__(self, num_regs, mem_size):
        self.ip = 0
        # self.registers = #initialize mem 0
        self.reg = [0 for i in range(num_regs)]
        # self.memory = # initialize mem 0
        self.mem = [0 for i in range(mem_size)]

    def execute(self, program):
        # Implement this one
        """Initialize Memory` value and Instruction List from program file, And Execute Program """
        self.mem = [
            int(mem_var)
            for mem_var in program[MEM_INDEX]
            .replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .split(", ")
        ]
        # Set up Instruction List from program assuming Size of inst_list is infinite
        self.inst_mem = []
        for index in range(MEM_INDEX + 1, len(program)):
            self.inst_mem.append(
                data[index]
                .replace("\n", "")
                .strip()
                .replace(",", "")
                .replace("$", "")
                .split(" ")
            )

            for size in range(OP_CODE + 1, len(self.inst_mem[index - 1])):
                self.inst_mem[index - 1][size] = int(self.inst_mem[index - 1][size])

        while self.ip < len(self.inst_mem):

            opcode = self.inst_mem[self.ip][OP_CODE]
            #    print(self.ip, self.mem, self.reg)
            if opcode == "ld":
                try:
                    self.reg[self.inst_mem[self.ip][MEM_DST_REG]] = self.mem[
                        self.inst_mem[self.ip][MEM_MEM]
                    ]
                    self.ip += 1
                except:
                    reg_e = self.inst_mem[self.ip][MEM_DST_REG]
                    mem_e = self.inst_mem[self.ip][MEM_MEM]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if reg_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg_e} | 0, {len(self.reg) -1 }"
                        )
                        exit(1)
                    elif mem_e > len(self.mem) - 1:
                        print(
                            f"SimpleCPU_MEMIndexError | IP:{self.ip} | {inst_e} | {mem_e} | 0, {len(self.mem) -1}"
                        )
                        exit(1)
            elif opcode == "st":
                try:
                    self.mem[self.inst_mem[self.ip][MEM_MEM]] = self.reg[
                        self.inst_mem[self.ip][MEM_SRC_REG]
                    ]
                    self.ip += 1
                except:
                    reg_e = self.inst_mem[self.ip][MEM_DST_REG]
                    mem_e = self.inst_mem[self.ip][MEM_MEM]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if reg_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg_e} | 0, {len(self.reg) -1 }"
                        )
                        exit(1)
                    elif mem_e > len(self.mem) - 1:
                        print(
                            f"SimpleCPU_MEMIndexError | IP:{self.ip} | {inst_e} | {mem_e} | 0, {len(self.mem) -1}"
                        )
                        exit(1)
            elif opcode == "add":
                try:
                    self.reg[self.inst_mem[self.ip][ARITH_DST]] = (
                        self.reg[self.inst_mem[self.ip][ARITH_SRC1]]
                        + self.reg[self.inst_mem[self.ip][ARITH_SRC2]]
                    )
                    self.ip += 1
                except:
                    dst_e = self.inst_mem[self.ip][ARITH_DST]
                    src1_e = self.inst_mem[self.ip][ARITH_SRC1]
                    src2_e = self.inst_mem[self.ip][ARITH_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if dst_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {dst_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src1_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src2_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
            elif opcode == "sub":
                try:
                    self.reg[self.inst_mem[self.ip][ARITH_DST]] = (
                        self.reg[self.inst_mem[self.ip][ARITH_SRC1]]
                        - self.reg[self.inst_mem[self.ip][ARITH_SRC2]]
                    )
                    self.ip += 1
                except:
                    dst_e = self.inst_mem[self.ip][ARITH_DST]
                    src1_e = self.inst_mem[self.ip][ARITH_SRC1]
                    src2_e = self.inst_mem[self.ip][ARITH_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if dst_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {dst_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src1_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src2_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
            elif opcode == "mul":
                try:
                    self.reg[self.inst_mem[self.ip][ARITH_DST]] = (
                        self.reg[self.inst_mem[self.ip][ARITH_SRC1]]
                        * self.reg[self.inst_mem[self.ip][ARITH_SRC2]]
                    )
                    self.ip += 1
                except:
                    dst_e = self.inst_mem[self.ip][ARITH_DST]
                    src1_e = self.inst_mem[self.ip][ARITH_SRC1]
                    src2_e = self.inst_mem[self.ip][ARITH_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if dst_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {dst_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src1_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src2_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
            elif opcode == "div":
                try:
                    self.reg[self.inst_mem[self.ip][ARITH_DST]] = (
                        self.reg[self.inst_mem[self.ip][ARITH_SRC1]]
                        / self.reg[self.inst_mem[self.ip][ARITH_SRC2]]
                    )
                    self.ip += 1
                except:
                    dst_e = self.inst_mem[self.ip][ARITH_DST]
                    src1_e = self.inst_mem[self.ip][ARITH_SRC1]
                    src2_e = self.inst_mem[self.ip][ARITH_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if dst_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {dst_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src1_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
                    elif src2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {src2_e} | 0, {len(self.reg)-1}"
                        )
                        exit(1)
            elif opcode == "jmp":
                try:
                    ip_check = self.ip
                    self.ip = self.inst_mem[self.ip][JUMP_INST_IDX]
                    ip_except_check = self.inst_mem[self.ip]
                    continue
                except:
                    inst_e = program[ip_check + 1].replace("\n", "")
                    print(
                        f"SimpleCPU_IPIndexError | IP:{ip_check} | {inst_e} | {self.ip} | 0, {len(self.inst_mem)-1}"
                    )
                    exit(1)
            elif opcode == "beq":
                try:
                    if (
                        self.reg[self.inst_mem[self.ip][BRAN_SRC1]]
                        == self.reg[self.inst_mem[self.ip][BRAN_SRC2]]
                    ):
                        ip_check = self.ip
                        self.ip = self.inst_mem[self.ip][BRAN_INST_IDX]
                        ip_except_check = self.inst_mem[self.ip]
                        continue
                    self.ip += 1
                except:
                    if self.ip > len(self.inst_mem) - 1:
                        inst_ip_e = program[ip_check + 1].replace("\n", "")
                        print(
                            f"SimpleCPU_IPIndexError | IP:{ip_check} | {inst_ip_e} | {self.ip} | 0 , {len(self.inst_mem)-1}"
                        )
                        exit(1)

                    reg1_e = self.inst_mem[self.ip][BRAN_SRC1]
                    reg2_e = self.inst_mem[self.ip][BRAN_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if reg1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg1_e} | 0, {len(self.reg) -1 }"
                        )
                        exit(1)
                    elif reg2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg2_e} | 0, {len(self.reg) -1}"
                        )
                        exit(1)
            elif opcode == "ble":
                try:
                    if (
                        self.reg[self.inst_mem[self.ip][BRAN_SRC1]]
                        <= self.reg[self.inst_mem[self.ip][BRAN_SRC2]]
                    ):
                        ip_check = self.ip
                        self.ip = self.inst_mem[self.ip][BRAN_INST_IDX]
                        ip_except_check = self.inst_mem[self.ip]
                        continue
                    self.ip += 1
                except:
                    if self.ip > len(self.inst_mem) - 1:
                        inst_ip_e = program[ip_check + 1].replace("\n", "")
                        print(
                            f"SimpleCPU_IPIndexError | IP:{ip_check} | {inst_ip_e} | {self.ip} | 0 , {len(self.inst_mem)-1}"
                        )
                        exit(1)

                    reg1_e = self.inst_mem[self.ip][BRAN_SRC1]
                    reg2_e = self.inst_mem[self.ip][BRAN_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if reg1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg1_e} | 0, {len(self.reg) -1 }"
                        )
                        exit(1)
                    elif reg2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg2_e} | 0, {len(self.reg) -1}"
                        )
                        exit(1)
            elif opcode == "bneq":
                try:
                    if (
                        self.reg[self.inst_mem[self.ip][BRAN_SRC1]]
                        != self.reg[self.inst_mem[self.ip][BRAN_SRC2]]
                    ):
                        ip_check = self.ip
                        self.ip = self.inst_mem[self.ip][BRAN_INST_IDX]
                        ip_except_check = self.inst_mem[self.ip]
                        continue
                    self.ip += 1
                except:
                    if self.ip > len(self.inst_mem) - 1:
                        inst_ip_e = program[ip_check + 1].replace("\n", "")
                        print(
                            f"SimpleCPU_IPIndexError | IP:{ip_check} | {inst_ip_e} | {self.ip} | 0 , {len(self.inst_mem)-1}"
                        )
                        exit(1)

                    reg1_e = self.inst_mem[self.ip][BRAN_SRC1]
                    reg2_e = self.inst_mem[self.ip][BRAN_SRC2]
                    inst_e = program[self.ip + 1].replace("\n", "")
                    if reg1_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg1_e} | 0, {len(self.reg) -1 }"
                        )
                        exit(1)
                    elif reg2_e > len(self.reg) - 1:
                        print(
                            f"SimpleCPU_REGIndexError | IP:{self.ip} | {inst_e} | {reg2_e} | 0, {len(self.reg) -1}"
                        )
                        exit(1)
        pass

    def print_status(self):
        # Implement this one
        print("IP:", self.ip)
        print("Register File:")
        for i in range(len(self.reg)):
            print(f"${i}: {self.reg[i]}")
        print("Memory:")
        for j in range(len(self.mem)):
            print(f"[{j}]: {self.mem[j]}")
        pass


# 상수 선언
NUM_REGS = 5
MEM_SZ = 10

MEM_INDEX = 0

OP_CODE = 0

ARITH_DST = 1
ARITH_SRC1 = 2
ARITH_SRC2 = 3

MEM_DST_REG = 1
MEM_SRC_REG = 1
MEM_MEM = 2

BRAN_SRC1 = 1
BRAN_SRC2 = 2
BRAN_INST_IDX = 3
JUMP_INST_IDX = 1

# 본 if문은 c의 main함수와 유사한 역할을 합니다.
# 프로그램을 파이썬 인터프리터로 실행할 경우에 실행되는 블락입니다.
# 다른 파이썬 프로그램이 본 파일을 라이브러리 형태로 참조하고자 할 경우 실행되지 않습니다.
if __name__ == "__main__":
    # sys.argv는 프로그램의 인자를 가진 리스트입니다.
    # sys.argv[0]은 여러분이 입력한 실행 프로그램의 이름이 들어있습니다.
    # sys.argv[1:] 부터 프로그램에 인자로 입력한 값이 들어 있습니다.
    if len(sys.argv) != 2:
        print(
            "<Usage>: ./emulator <program.txt>"
        )  # program이나py파일이 하나라도 안들어왔을 경우 exception 처리
        exit(1)
    with open(sys.argv[1]) as f:
        # 텍스트 파일의 데이터를 전부 읽어오는 코드 입니다.
        # 이 부분은 for loop 등 기타 방법으로 변형해서 사용해도 됩니다.
        data = f.readlines()
    # 이 부분은 예시로 제공된 코드입니다. 실제로 동작하지 않으니 동작하도록 수정해서 사용해야 합니다.
    cpu = SimpleCPU(NUM_REGS, MEM_SZ)
    cpu.execute(data)
    cpu.print_status()
