import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load():
    x = pd.read_csv('logisticX.csv', header=None).values
    y = pd.read_csv('logisticY.csv', header=None).values
    return x, y

def norm(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std

def sig(z):
    return 1 / (1 + np.exp(-z))

def cost(x, y, theta):
    m = len(y)
    h = sig(x.dot(theta))
    c = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return c

def grad_desc(x, y, theta, alpha, iters):
    m = len(y)
    cost_hist = []
    for _ in range(iters):
        h = sig(x.dot(theta))
        grad = (1/m) * x.T.dot(h - y)
        theta -= alpha * grad
        cost_hist.append(cost(x, y, theta))
    return theta, cost_hist

def plot_cost(cost_hist, label):
    plt.plot(range(len(cost_hist)), cost_hist, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.legend()

def plot_boundary(x, y, theta):
    plt.figure()
    cls_0 = (y.flatten() == 0)
    cls_1 = (y.flatten() == 1)
    plt.plot(x[cls_0, 1], x[cls_0, 2], 'r', label='Class 0')
    plt.plot(x[cls_1, 1], x[cls_1, 2], 'b', label='Class 1')
    x_vals = np.array([min(x[:, 1]), max(x[:, 1])])
    y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]
    plt.plot(x_vals, y_vals, 'g-', label='Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Decision Boundary')
    plt.show()

def eval(x, y, theta):
    y_pred = sig(x.dot(theta)) >= 0.5
    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    cm = np.array([[tn, fp], [fn, tp]])
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) != 0 else 0
    rec = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    return cm, acc, prec, rec, f1

# Workflow
x, y = load()
x = norm(x)
x = np.c_[np.ones((x.shape[0], 1)), x]
theta = np.zeros((x.shape[1], 1))
alpha = 0.1
iters = 1000

# Training
theta_opt, cost_hist = grad_desc(x, y, theta, alpha, iters)

# Print final cost and optimal theta
final_cost = cost(x, y, theta_opt)
print(f'Final Cost: {final_cost:.4f}')
print(f'Optimal Theta: {theta_opt.flatten()}')

# Plot cost vs iterations
plot_cost(cost_hist, label=f'Learning Rate {alpha}')
plt.show()

# Plot decision boundary
plot_boundary(x, y, theta_opt)

# Model evaluation
cm, acc, prec, rec, f1 = eval(x, y, theta_opt)
print('Confusion Matrix:\n', cm)
print(f'Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-score: {f1:.2f}')

# Compare different learning rates
alpha_2 = 5
iters_short = 100
theta_init = np.zeros((x.shape[1], 1))
_, cost_hist_2 = grad_desc(x, y, theta_init, alpha_2, iters_short)

plt.figure()
plot_cost(cost_hist[:iters_short], label=f'Learning Rate {alpha}')
plot_cost(cost_hist_2, label=f'Learning Rate {alpha_2}')
plt.title('Cost Comparison for Different Learning Rates')
plt.show()
