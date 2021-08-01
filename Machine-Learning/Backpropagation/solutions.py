import numpy as np

"""
Debería resolver esta práctica sin agregar más librerías externas
"""

def NotImplemented_message():
    print('###################################')
    print('Tienen que implementar esta función')
    print('###################################')
    return np.array([1, 1])

def densa_forward(X, W, b):
    return np.dot(X,W)+b

def MSE(X_true, X_pred):
    return ((X_true-X_pred)**2).sum()/len(X_true[0])

def MSE_grad(X_true, X_pred):
    return (2*(X_pred-X_true)/len(X_true[0])).T

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_jac(Xin):
    sigmoid_out = sigmoid(Xin)*(1-sigmoid(Xin))
    return np.diag(sigmoid_out.reshape(-1))

def softmax(z):
    return np.exp(z)/np.exp(z).sum()

def softmax_jac(Xin):
    softmax_out = softmax(Xin)
    return np.diag(softmax_out.reshape(-1))-softmax_out.T.dot(softmax_out)

def forward(X, P_true, weights):
    D1_out = densa_forward(X, weights[0], weights[1])
    A1_out = sigmoid(D1_out)
    D2_out = densa_forward(A1_out, weights[2], weights[3])
    A2_out = sigmoid(D2_out)
    D3_out = densa_forward(A2_out, weights[4], weights[5])
    P_est = softmax(D3_out)
    mse = MSE(P_est, P_true)
    return P_est, mse, X, A1_out, A2_out

def get_gradients(X, P_true, weights):
    D1_out = densa_forward(X, weights[0], weights[1])
    A1_out = sigmoid(D1_out)
    D2_out = densa_forward(A1_out, weights[2], weights[3])
    A2_out = sigmoid(D2_out)
    D3_out = densa_forward(A2_out, weights[4], weights[5])
    P_est = softmax(D3_out)
    MSE_grad_out = MSE_grad(P_true, P_est)
    error_D3 = softmax_jac(D3_out).dot(MSE_grad_out)
    error_A2 = weights[4].dot(error_D3)
    error_D2 = sigmoid_jac(D2_out).dot(error_A2)
    error_A1 = weights[2].dot(error_D2)
    error_D1 = sigmoid_jac(D1_out).dot(error_A1)
    g_1_ws = np.matmul(error_D1, X).T
    g_1_b = np.matmul(error_D1, np.array([[1]])).T
    g_2_ws = np.matmul(error_D2, A1_out).T
    g_2_b = np.matmul(error_D2, np.array([[1]])).T
    g_3_ws = np.matmul(error_D3, A2_out).T
    g_3_b = np.matmul(error_D3, np.array([[1]])).T
    return tuple([g_1_ws, g_1_b, g_2_ws, g_2_b, g_3_ws, g_3_b]), MSE(P_true, P_est)