from gekko import GEKKO
import numpy as np
import time

def achieved_acc(avg_accuracies=None, micro_batch_sizes=None):
    assert(len(avg_accuracies) == len(micro_batch_sizes))
    total = 0
    for i, avg_acc in enumerate(avg_accuracies):
        total += micro_batch_sizes[i] * avg_acc

    return total / np.sum(micro_batch_sizes)

def solve(
        batch_size=None,
        max_batch_size=None,
        num_mod_policies=1, 
        avg_accuracies=None,
        acc_target=None,
        metrics=None):

    avg_accuracies = np.array(avg_accuracies)
    m = GEKKO(remote=False)

    # index decision variable
    x = m.Array(m.Var,num_mod_policies,value=num_mod_policies+1,lb=0,
            ub=batch_size,integer=True)

    # cublic spline
    ix = np.array([i for i in range(0, max_batch_size+1)])
    lists = np.array([m for m in metrics])
    y = m.Array(m.Var,num_mod_policies)
    
    for i,iy in enumerate(lists):
        m.cspline(x[i], y[i], ix, iy, bound_x=True)
    
    # constraints
    m.Equations([m.sum(x)==batch_size,
                 m.sum(np.multiply(avg_accuracies, x)) >= acc_target * batch_size])
    
    # objective
    m.Minimize(m.sum(y))

    # solve
    # m.options.SOLVER = 1
    m.options.SOLVER = 3
    # m.options.DIAGLEVEL = 1
    # m.options.IMODE = 3
    m.options.IMODE = 2
    # m.options.TIME_SHIFT=0
    m.options.REDUCE=3
    m.options.WEB = 0

    start = time.time()
    m.solve(disp=False)
    end = time.time()

    # print("lat: ", end - start)
    # print('Solver Time 2: ', m.options.SOLVETIME)

    # print(f'x: {x}')
    # print(f'y: {y}')
    # print(f'objective: {m.options.OBJFCNVAL}')

    # process x and y
    x = np.array([x_i[0] for x_i in x])
    y = np.array([y_i[0] for y_i in y])

    # get achieved accuracy
    actual_acc = achieved_acc(avg_accuracies, x) 

    return (x,
            y,
            m.options.OBJFCNVAL,
            acc_target,
            actual_acc,
            m.options.SOLVETIME,
            end-start)

def queue_solve(
        latencies,
        accuracies,
        time_budget,
    ):
    m = GEKKO(remote=False)

    # index decision variables into the database
    assert(len(latencies) == len(accuracies))
    index_decisions = [m.Var(0, lb=0, ub=len(l)-1, integer=True) \
            for l in latencies]
    
    ix = []
    for b, l in enumerate(latencies):
        assert(len(latencies[b]) == len(accuracies[b]))
        length = len(l)
        ix.append([list(range(0, length))])

    # accuracies
    num_jobs = len(accuracies)
    accs = m.Array(m.Var, num_jobs)
    for i, ia in enumerate(accuracies):
        m.cspline(index_decisions[i], accs[i], ix[i], ia, bound_x=True)

    # latencies
    lats = m.Array(m.Var, num_jobs)
    for i, il in enumerate(latencies):
        m.cspline(index_decisions[i], lats[i], ix[i], il, bound_x=True)

    # constraints
    m.Equations([
        m.sum(lats) <= time_budget,
        ])
    
    # obj
    avg_acc = m.Intermediate(m.sum(accs) / num_jobs)
    m.Minimize(-avg_acc)

    # solve
    # m.options.SOLVER = 1
    m.options.SOLVER = 3
    # m.options.DIAGLEVEL = 1
    # m.options.IMODE = 3
    m.options.IMODE = 2
    # m.options.TIME_SHIFT=0
    m.options.REDUCE=3
    m.options.WEB = 0

    start = time.time()
    try:
        m.solve(disp=False)
    except:
        print("inside queue solver\n")
        print("latencies: ", latencies)
        print("acc: ", accuracies)
        print("time budget: ", time_budget)
        raise Exception("Solution not found")
    end = time.time()

    # process x and y
    index_decisions = np.array([int(i[0]) for i in index_decisions])
    accs = np.array([a[0] for a in accs])
    lats = np.array([l[0] for l in lats])

    return index_decisions, accs, lats, m.options.OBJFCNVAL, \
            m.options.SOLVETIME, end-start


