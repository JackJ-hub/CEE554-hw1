import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def pla(x, y, max_iter=1000):
    N = x.shape[0]
    xb = np.c_[np.ones(N), x]
    w = np.zeros(xb.shape[1])
    Ein = []
    updates = 0

    for j in range(max_iter):
        changed = False
        for i in range(N):
            if y[i] * (w @ xb[i]) <= 0:
                w += y[i] * xb[i]
                updates += 1
                changed = True

        y_hat = np.where((xb @ w) > 0, 1, -1)
        Ein.append(np.mean(y_hat != y))

        if not changed:
            print(f'Converged after {updates} updates.')
            break

    return w, updates, np.array(Ein)


def plot_scatter_with_boundary(x, y, w):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[y==1, 0], x[y==1, 1], c="red", marker="o", label="y=+1")
    ax.scatter(x[y==-1, 0], x[y==-1, 1], c="green", marker="x", label="y=-1")

    w0, w1, w2 = w
    xs = np.linspace(x[:,0].min(), x[:,0].max(), 200)
    if abs(w2) > 1e-12:
        ax.plot(xs, -(w0 + w1*xs)/w2, linewidth=2, label="boundary")
    elif abs(w1) > 1e-12:
        ax.axvline(-w0/w1, linewidth=2, label="boundary")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()

def shuffle_experiment(x, y, n_runs=10, seed=0, max_iter=1000):
    rng = np.random.default_rng(seed)
    updates_list = []

    for r in range(n_runs):
        idx = rng.permutation(len(y)) #learned from AI that how to shuffle randomly and repeatably
        _, updates, _ = pla(x[idx], y[idx], max_iter=max_iter)
        updates_list.append(updates)
        

    return updates_list

def pla_T_updates(x, y, T=1000):
    N = x.shape[0]
    xb = np.c_[np.ones(N), x]
    w = np.zeros(xb.shape[1])

    Ein = []
    updates = 0

    while updates < T:
        changed = False
        for i in range(N):
            if y[i] * (w @ xb[i]) <= 0:
                w += y[i] * xb[i]
                updates += 1
                changed = True

                y_hat = np.where((xb @ w) > 0, 1, -1)
                Ein.append(np.mean(y_hat != y))

                if updates >= T:
                    break

        if not changed:
            print(f'Converged after {updates} updates.')
            break

    return w, updates, np.array(Ein)

def pocket_pla_T_updates(x, y, T=1000):
    N = x.shape[0]
    xb = np.c_[np.ones(N), x]
    w = np.zeros(xb.shape[1])

    Ein = []
    updates = 0
    w_best = w.copy()
    y_hat = np.where((xb @ w_best) > 0, 1, -1)
    
    min_Ein = np.mean(y_hat != y)

    while updates < T:
        changed = False
        for i in range(N):
            if y[i] * (w @ xb[i]) <= 0:
                w += y[i] * xb[i]
                updates += 1
                changed = True

                y_hat = np.where((xb @ w) > 0, 1, -1)
                Ein_count = np.mean(y_hat != y)
                Ein.append(Ein_count)
                if Ein_count < min_Ein:
                    min_Ein = Ein_count
                    w_best = w.copy()

                if updates >= T:
                    break

        if not changed:
            print(f'Converged after {updates} updates.')
            break

    return w, updates, np.array(Ein), w_best, min_Ein

def plot_two_boundaries(x, y, wT, wbest):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[y==1, 0], x[y==1, 1], c="red", label="y=+1")
    ax.scatter(x[y==-1, 0], x[y==-1, 1], c="green", label="y=-1")

    xs = np.linspace(x[:,0].min(), x[:,0].max(), 200)

    def draw(w, label):
        w0, w1, w2 = w
        if abs(w2) > 1e-12:
            ax.plot(xs, -(w0 + w1*xs)/w2, linewidth=2, label=label)
        elif abs(w1) > 1e-12:
            ax.axvline(-w0/w1, linewidth=2, label=label)

    draw(wT, "w(T)")
    draw(wbest, "wbest")

    ax.set_xlabel("x1");ax.set_ylabel("x2");ax.grid(alpha=0.3);ax.legend();plt.show()


def main():
    path = Path("/Users/jack/Desktop/CEE554/HW1/Bridge_Condition.txt")
    data = np.loadtxt(path) #delimiter & skiprows
    #print(data) 
    dataset1 = data[0:20,:]
    dataset2 = data[:,:]
    # Q1: Scatter plot of dataset1
    x1 = dataset1[:,0:2]
    y1 = dataset1[:,2]

    plt.figure(figsize=(8,6))
    plt.scatter(x1[y1==1, 0],  x1[y1==1, 1],  c='red', label='y = +1 (red)') # learned form AI that how to plot without loop
    plt.scatter(x1[y1==-1, 0], x1[y1==-1, 1], c='green', label='y = -1 (green)')
    plt.xlabel('x1');plt.ylabel('x2');plt.title('Scatter Plot of Dataset2');plt.grid(alpha=0.3);plt.legend();plt.show()

    # Q1b
    # (b) Run PLA on Dataset 1 (no shuffle, file order)
    w_b, updates_b, Ein_b = pla(x1, y1)

    print(f"(b-i) updates until convergence = {updates_b}")

    # b-ii
    plot_scatter_with_boundary(x1, y1, w_b)

    # b-iii
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(1, len(Ein_b) + 1), Ein_b, linewidth=2)
    plt.xlabel("Iteration (pass/epoch)")
    plt.ylabel("Ein")
    plt.grid(alpha=0.3)
    plt.show()
    #Q1c
    updates_list = shuffle_experiment(x1, y1, n_runs=10, seed=0, max_iter=1000)

    print("\nUpdates to convergence for each shuffle:")
    print(updates_list)

    # Q1d
    x = dataset2[:,0:2]
    y = dataset2[:,2]

    plt.figure(figsize=(8,6))
    plt.scatter(x[y==1, 0],  x[y==1, 1],  c='red', label='y = +1 (red)') # learned form AI that how to plot without loop
    plt.scatter(x[y==-1, 0], x[y==-1, 1], c='green', label='y = -1 (green)')
    plt.xlabel('x1');plt.ylabel('x2');plt.title('Scatter Plot of Dataset2');plt.grid(alpha=0.3);plt.legend();plt.show()

    #1e
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    xs, ys = x[idx], y[idx]
    w, updates, Ein_e = pla_T_updates(xs, ys, T=1000)
    iters = np.arange(1, len(Ein_e) + 1)
    plt.figure(figsize=(8,4))
    plt.plot(iters, Ein_e, linewidth=2)
    plt.xlabel("Iteration (update number)")
    plt.ylabel("Misclassification rate")
    plt.grid(alpha=0.3)
    plt.show()


    wT, updates, Ein, wbest, min_Ein = pocket_pla_T_updates(xs, ys, T=1000)

    print("Best (lowest) misclassification rate =", min_Ein)
    print("wbest =", wbest)
    print("w(T) =", wT)
    plot_two_boundaries(xs, ys, wT, wbest)






if __name__ == "__main__":
    main()

    
