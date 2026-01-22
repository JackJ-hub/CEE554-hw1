import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# =========================
# ========== PLA ==========
# =========================
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


# ==============================
# ========== Plotting ==========
# ==============================
def plot_scatter_with_boundary(x, y, w, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c="red", marker="o", label="y=+1")
    ax.scatter(x[y == -1, 0], x[y == -1, 1], c="green", marker="x", label="y=-1")

    w0, w1, w2 = w
    xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
    if abs(w2) > 1e-12:
        ax.plot(xs, -(w0 + w1 * xs) / w2, linewidth=2, label="boundary")
    elif abs(w1) > 1e-12:
        ax.axvline(-w0 / w1, linewidth=2, label="boundary")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()


def plot_two_boundaries(x, y, wA, wB, labelA="A", labelB="B", title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c="red", marker="o", label="y=+1")
    ax.scatter(x[y == -1, 0], x[y == -1, 1], c="green", marker="x", label="y=-1")

    xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)

    def draw(w, label):
        w0, w1, w2 = w
        if abs(w2) > 1e-12:
            ax.plot(xs, -(w0 + w1 * xs) / w2, linewidth=2, label=label)
        elif abs(w1) > 1e-12:
            ax.axvline(-w0 / w1, linewidth=2, label=label)

    draw(wA, labelA)
    draw(wB, labelB)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()


def plot_side_by_side(x, y, w_left, w_right, left_name="Left", right_name="Right", title=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def draw(ax, w, name):
        ax.scatter(x[y == 1, 0], x[y == 1, 1], c="red", marker="o", label="y=+1")
        ax.scatter(x[y == -1, 0], x[y == -1, 1], c="green", marker="x", label="y=-1")
        xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
        w0, w1, w2 = w
        if abs(w2) > 1e-12:
            ax.plot(xs, -(w0 + w1 * xs) / w2, linewidth=2, label="boundary")
        elif abs(w1) > 1e-12:
            ax.axvline(-w0 / w1, linewidth=2, label="boundary")
        ax.set_title(name)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.3)
        ax.legend()

    draw(axes[0], w_left, left_name)
    draw(axes[1], w_right, right_name)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ==========================================
# ========== Logistic-MSE (SGD) ============
# ==========================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def rho(z):
    return 2.0 * sigmoid(z) - 1.0


def logistic_mse_cost(w, xb, y):
    r = rho(xb @ w)
    return np.mean((r - y) ** 2)


def logistic_mse_error(w, xb, y):
    y_hat = np.where((xb @ w) > 0, 1, -1)
    return np.mean(y_hat != y)


def logistic_mse_sgd(x, y, lr=0.1, T=5000, seed=0, w0=None, record_every=1):
    """
    SGD 每次迭代用一个随机点更新一次 w（一次 update 记为一次 iteration）
    返回:
      w_final,
      history_cost (len ~= T/record_every),
      history_err  (len ~= T/record_every)
    """
    rng = np.random.default_rng(seed)
    N = x.shape[0]
    xb = np.c_[np.ones(N), x]

    if w0 is None:
        w = rng.normal(0, 1, size=xb.shape[1])
    else:
        w = w0.astype(float).copy()

    cost_hist = []
    err_hist = []

    for t in range(1, T + 1):
        i = rng.integers(0, N)
        xi = xb[i]
        yi = y[i]

        z = float(w @ xi)
        s = sigmoid(z)
        r = 2.0 * s - 1.0

        # grad = 4 (rho - y) sigma(1-sigma) x
        grad = 4.0 * (r - yi) * s * (1.0 - s) * xi
        w -= lr * grad

        if (t % record_every) == 0:
            cost_hist.append(logistic_mse_cost(w, xb, y))
            err_hist.append(logistic_mse_error(w, xb, y))

    return w, np.array(cost_hist), np.array(err_hist)


def logistic_mse_multi_restart(x, y, restarts=20, lr=0.1, T=20000, seed=0, record_every=10):
    """
    多次随机初始化，挑最好的（优先 min final error，再看 final cost）
    """
    best = None
    best_pack = None

    for r in range(restarts):
        w, cost_hist, err_hist = logistic_mse_sgd(
            x, y, lr=lr, T=T, seed=seed + r, w0=None, record_every=record_every
        )
        final_err = err_hist[-1]
        final_cost = cost_hist[-1]
        key = (final_err, final_cost)

        if (best is None) or (key < best):
            best = key
            best_pack = (w, cost_hist, err_hist)

    w_best, cost_best, err_best = best_pack
    return w_best, cost_best, err_best, {"final_err": best[0], "final_cost": best[1]}


def plot_cost_err(cost_hist, err_hist, title_prefix="Logistic-MSE"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(cost_hist, linewidth=2)
    axes[0].set_title(f"{title_prefix}: Cost")
    axes[0].set_xlabel("record index")
    axes[0].set_ylabel("C(w)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(err_hist, linewidth=2)
    axes[1].set_title(f"{title_prefix}: Misclassification Rate")
    axes[1].set_xlabel("record index")
    axes[1].set_ylabel("Ein")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =========================
# ========== Main =========
# =========================
def main():
    path = Path("/Users/jack/Desktop/CEE554/HW1/Bridge_Condition.txt")
    data = np.loadtxt(path)

    dataset1 = data[0:20, :]
    dataset2 = data[:, :]

    # -------------------------
    # Dataset 1
    # -------------------------
    x1 = dataset1[:, 0:2]
    y1 = dataset1[:, 2].astype(int)

    plt.figure(figsize=(8, 6))
    plt.scatter(x1[y1 == 1, 0], x1[y1 == 1, 1], c='red', label='y=+1')
    plt.scatter(x1[y1 == -1, 0], x1[y1 == -1, 1], c='green', label='y=-1')
    plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Scatter Plot of Dataset 1')
    plt.grid(alpha=0.3); plt.legend(); plt.show()

    # PLA on Dataset 1
    w_pla1, updates_pla1, Ein_pla1 = pla(x1, y1)
    print(f"[Dataset 1] PLA converged updates = {updates_pla1}")
    plot_scatter_with_boundary(x1, y1, w_pla1, title="Dataset 1: PLA final classifier")

    # Logistic-MSE on Dataset 1 (multi-restart)
    w_log1, cost1, err1, info1 = logistic_mse_multi_restart(
        x1, y1,
        restarts=30,      # 你可以加大
        lr=0.1,           # 你可以调
        T=3000,          # 你可以加大
        seed=0,
        record_every=10   # 每 10 次 update 记录一次（不然记录太密）
    )
    print(f"[Dataset 1] Logistic-MSE best final_err={info1['final_err']:.4f}, final_cost={info1['final_cost']:.6f}")
    plot_scatter_with_boundary(x1, y1, w_log1, title="Dataset 1: Logistic-MSE final classifier")
    plot_cost_err(cost1, err1, title_prefix="Dataset 1 Logistic-MSE")

    # Side-by-side: PLA vs Logistic-MSE (Dataset 1)
    plot_side_by_side(
        x1, y1, w_pla1, w_log1,
        left_name="PLA (Problem 1)", right_name="Logistic-MSE (SGD)",
        title="Dataset 1: PLA vs Logistic-MSE"
    )

    # -------------------------
    # Dataset 2
    # -------------------------
    x2 = dataset2[:, 0:2]
    y2 = dataset2[:, 2].astype(int)

    plt.figure(figsize=(8, 6))
    plt.scatter(x2[y2 == 1, 0], x2[y2 == 1, 1], c='red', label='y=+1')
    plt.scatter(x2[y2 == -1, 0], x2[y2 == -1, 1], c='green', label='y=-1')
    plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Scatter Plot of Dataset 2')
    plt.grid(alpha=0.3); plt.legend(); plt.show()

    # Pocket PLA (Problem 1) on Dataset 2
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y2))
    xs, ys = x2[idx], y2[idx]

    wT2, updates2, Ein2, wbest2, min_Ein2 = pocket_pla_T_updates(xs, ys, T=1000)
    print(f"[Dataset 2] Pocket PLA best Ein = {min_Ein2:.4f}")
    print("[Dataset 2] wbest =", wbest2)
    print("[Dataset 2] w(T)  =", wT2)

    # Logistic-MSE on Dataset 2 (multi-restart)
    w_log2, cost2, err2, info2 = logistic_mse_multi_restart(
        xs, ys,
        restarts=40,
        lr=0.1,
        T=60000,
        seed=10,
        record_every=20
    )
    print(f"[Dataset 2] Logistic-MSE best final_err={info2['final_err']:.4f}, final_cost={info2['final_cost']:.6f}")

    plot_scatter_with_boundary(xs, ys, w_log2, title="Dataset 2: Logistic-MSE final classifier")
    plot_cost_err(cost2, err2, title_prefix="Dataset 2 Logistic-MSE")

    # Side-by-side: Pocket PLA vs Logistic-MSE (Dataset 2)
    plot_side_by_side(
        xs, ys, wbest2, w_log2,
        left_name="Pocket PLA (Problem 1 wbest)", right_name="Logistic-MSE (SGD)",
        title="Dataset 2: Pocket PLA vs Logistic-MSE"
    )


if __name__ == "__main__":
    main()
