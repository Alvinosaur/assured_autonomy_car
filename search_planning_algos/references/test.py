import queue
from threading import Thread
from copy import deepcopy


class A():
    def __init__(self, x):
        self.x = x


def foo(args):
    a, i = args
    print('hello {0}'.format(a.x))
    a.x = i
    return "foo {}".format(a.x)


def bar(i):
    print("bar %d" % i)


que = queue.Queue()
threads_list = list()

iters = 20
for i in range(iters):
    a = A(0)
    t = Thread(target=lambda q, args: q.put(
        foo(args)), args=(que, (a, i)))
    t.start()
    threads_list.append(t)

# Join all the threads
for t in threads_list:
    t.join()

# Check thread's return value
while not que.empty():
    result = que.get()
    print(result)
