#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import timeit, math, random, sortedcontainers
from collections import deque
from copy import deepcopy
from numpy.lib.utils import deprecate_with_doc


class PriorityQueue(object): # cited form hw1
    def __init__(self, node):
        self._queue = sortedcontainers.SortedList([node])
    def push(self, node):
        self._queue.add(node)
    def pop(self):
        return self._queue.pop(index=0)
    def empty(self):
        return len(self._queue) == 0
    def compare_and_replace(self, i, node):
        if node < self._queue[i]:
            self._queue.pop(index=i)
            self._queue.add(node)
    def find(self, node):
        try:
            loc = self._queue.index(node)
            return loc
        except ValueError:
            return None


class Node(object):  # Represents a node(chessboard) in a search tree
    # state是nparray类型的二维矩阵，存有棋盘的值
    # action是形如(turns,(x1,y1),(x2,y2))的tuple，指明一种消法。规定x小的在前，一样时y小的在前
    def __init__(self, state, turn_count=0, parent=None):
        self.state = state
        self.parent = parent
        self.turn_count = turn_count # 棋盘变到此状态的累计的转弯损失，每个直连消除损失1，每个拐n次消除损失n+1。该值相当于g
        self.renew_cost()
    
    def renew_cost(self): #利用state算出turn_future，再结合turn_count算出path_cost
        self.turn_future = int(sum(sum(self.state>0))/2) # state中正值(图案)的总个数/2，作为h
        self.path_cost = self.turn_count+self.turn_future # f=g+h
        return self.path_cost

    def child_node(self, action):
        next_state = deepcopy(self.state) # 重要！必须deepcopy才不会修改原state
        next_state[action[1][0]][action[1][1]] = 0
        next_state[action[2][0]][action[2][1]] = 0        
        next_turn_count = self.turn_count+action[0]+1 #孩子的turn_count是父亲的+本次消除的，action[0]要加1是因为直连损失1，拐n次损失n+1
        next_node = Node(next_state, next_turn_count, self)
        return next_node

    def path(self): # Returns list of nodes from the root node to this node
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return "<Node {}(path_cost={},turn_count={})>".format(self.state, self.path_cost, self.turn_count)
    def __lt__(self, other):
        return self.path_cost < other.path_cost
    def __eq__(self, other):
        cmp = np.equal(self.state, other.state)
        return cmp.all() # nparray相等比较，需要调用all()，这样只有所有元素全等才等


class LLKProblem(object):
    def __init__(self, mode=2, m=6, n=6, k=18, p=11, z=0):
        #默认自动构造。若要手动构造，则创建实例后再修改self.init_node
        self.mode = mode
        self.m = m
        self.n = n
        self.k = k
        self.p = p
        self.z = z

        if k>16:
            self.maxbranch=1
        elif k>11:
            self.maxbranch=2
        elif k>8 or (m*n)>40:
            self.maxbranch=3
        else:
            self.maxbranch=4
        
        self.maxturn = 2 if (mode==1) else 3

        self.init_state = self.initialize()
        # self.init_state=np.array([[0, 0, 0, 0, 0, 0, 0, 0],
        #                     [0, 8, 9, 1, 2, 11, 8, 0],
        #                     [0, 11, 9, 2, 1, 9, 11, 0],
        #                     [0, 12, 6, 1, 2, 5, 3, 0],
        #                     [0, 5, 7, 2, 1, 12, 6, 0],
        #                     [0, 11, 9, 3, 4, 10, 11, 0],
        #                     [0, 10, 11, 4, 7, 11, 11, 0],
        #                     [0, 0, 0, 0, 0, 0, 0, 0]])

        self.goal_state = np.zeros((m+2,n+2))
        self.init_node = Node(self.init_state)
        self.goal_node = Node(self.goal_state)
        self.display_state = self.init_state #用于图形界面展示，无关算法

    
    def initialize(self):
        init_state = np.zeros((self.m+2, self.n+2)) #四周被一圈0环绕，0表示空位，可用于连线
        locations = random.sample(range(0, self.m*self.n), 2*self.k+self.z) #随机2k+z个位置
        for i in range(self.k): #放置图案(正数)
            type = (i%self.p)+1
            row = math.floor(locations[2*i]/self.n)
            col = locations[2*i] - row*self.n
            init_state[row+1][col+1] = type
            row = math.floor(locations[2*i+1]/self.n)
            col = locations[2*i+1] - row*self.n
            init_state[row+1][col+1] = type
        for i in range(2*self.k, 2*self.k+self.z): #放置阻挡(-1)
            row = math.floor(locations[i]/self.n)
            col = locations[i] - row*self.n
            init_state[row+1][col+1] = -1
        return init_state

    def actions(self, state):#mode为1只允许两拐及以下，mode为2或3允许多拐
        actions_set = set() #使用set避免重复
        rows,cols = state.shape
        minimum_turns = 5
        for start_i in range(rows):
            for start_j in range(cols):
                if len(actions_set)>=self.maxbranch: #控制分支总量                    
                    return list(actions_set)
                if state[start_i][start_j]<=0:
                    continue
                pictype = state[start_i][start_j]
                #寻找与其同类的pic

                turns_matrix = -3*np.ones((rows,cols))
                #-3表示unsearched,-2表示障碍物或不一致的图,-1表示起始点，非负表示到达该位置拐几下
                turns_matrix[start_i][start_j] = -1

                current_turns = 0 # 首先找0次拐弯的点
                breakflag = False
                while True:
                    #从拐current_turns-1次的点出发，找到拐current_turns的点，不断增加current_turns
                    if (current_turns>self.maxturn) or (current_turns>(minimum_turns+1)): #超过预设的最大拐数，或超过本次最小拐+1，则忽略
                        break
                    startpointlst = np.where(turns_matrix==(current_turns-1)) #startpoint[0]是行坐标list, startpoint[1]是列坐标list
                    for sp in range(len(startpointlst[0])):
                        i = startpointlst[0][sp]
                        j = startpointlst[1][sp]
                        #从(i, j)出发，往上下左右寻找尚未被发现的直连点                        
                        for dir in range(4):
                            k = 1
                            #dir=0 左；dir=1 上；dir=2 右；dir=3 下；
                            cur_i = i+(dir%2)*(dir-2)*k
                            cur_j = j+((dir+1)%2)*(dir-1)*k
                            while 0<=cur_i and cur_i<rows and 0<=cur_j and cur_j<cols:
                                if state[cur_i][cur_j]==0 and turns_matrix[cur_i][cur_j]==-3: #空位置，且之前没探索过
                                    turns_matrix[cur_i][cur_j]=current_turns #拐current_turns可达
                                elif state[cur_i][cur_j]==pictype and turns_matrix[cur_i][cur_j]==-3: #找到了
                                    #让两坐标点以固定顺序排列在action里，相互查找时获得一样的action，便于去重
                                    judge = start_i<cur_i or (start_i==cur_i and start_j<cur_j)
                                    first_i = start_i if judge else cur_i
                                    first_j = start_j if judge else cur_j
                                    second_i = cur_i if judge else start_i
                                    second_j = cur_j if judge else start_j
                                    action = (current_turns,(first_i,first_j),(second_i,second_j))
                                    actions_set.add(action)
                                    minimum_turns = current_turns if current_turns<minimum_turns else minimum_turns                                   
                                    breakflag = True
                                    break
                                elif state[cur_i][cur_j]==0 or state[cur_i][cur_j]==pictype: #已探索过的空位置或起点，啥也不干
                                    pass
                                else: #遇到障碍 或 不一致的图
                                    turns_matrix[cur_i][cur_j]=-2
                                    break
                                if breakflag:
                                    break
                                k = k + 1
                                cur_i = i+(dir%2)*(dir-2)*k
                                cur_j = j+((dir+1)%2)*(dir-1)*k                            
                            if breakflag:
                                break
                        if breakflag:
                            break
                    if breakflag:
                        break
                    current_turns = current_turns+1

        return list(actions_set)

    def expand(self, node):  # Returns a list of child nodes
        return [node.child_node(action) for action in self.actions(node.state)] 

    
def search(problem):
    # 使用A*    
    PQ = PriorityQueue(problem.init_node)
    old_node = problem.init_node #赋值无意义，只为提前声明old_node
    while not PQ.empty():
        old_node = PQ.pop()
        if (old_node==problem.goal_node):
            return old_node.path() #连光了
        for child_node in problem.expand(old_node):            
            PQ.push(child_node)

    return old_node.path() #没连光，没得可连了
    

if __name__ == "__main__":
    problem = LLKProblem()    
    start=timeit.default_timer()
    print("\n使用A*求解LLK，%d行%d列，%d种%d图案，%d障碍，最大分支%d，最多拐%d次："\
        %(problem.m, problem.n, problem.p, 2*problem.k, problem.z, problem.maxbranch, problem.maxturn))
    lst = search(problem)

    if len(lst)>20:
        print(lst[0])
        print("\n共%d步，略去中间步骤\n"%(len(lst)-1))
        print(lst[len(lst)-1])
    else:
        for i in lst:
            print(i)
            print("\n")   

    end=timeit.default_timer()
    print('\nRunning time: %.5s Seconds'%(end-start))
