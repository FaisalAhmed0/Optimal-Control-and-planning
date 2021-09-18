import gym
import env
import matplotlib.pyplot as plt
from autograd import grad, jacobian
import autograd.numpy as np
from PIL import Image
import warnings


# https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
class ILQR():
    def __init__(self, dynamics, running_cost, final_cost, state_shape, action_shape, horizon, mu_max):
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.final_cost = final_cost


        self.horizon = horizon  
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.v = np.array([0.0 for _ in range(self.horizon)])

        self.mu_min = 1e-6
        self.mu = 1.0
        self.mu_max = mu_max
        self.delta_0 = 2.0
        # Backtracking Line search parameter
        self.alpha = 1
        self.gamma = 0.5
        # The requried derivatives
        # Costs dervivatives 0-> with respict to x and 1-> with repect to y
        self.l_x = grad(running_cost, 0)
        self.l_u = grad(running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        # Dynamics Derivative
        self.f_x = jacobian(dynamics, 0)
        self.f_u = jacobian(dynamics, 1)
        # final cost derivatives
        self.l_f_x = grad(final_cost)
        self.l_f_xx= jacobian(self.l_f_x)
        # Values matrices
        self.V_XX = [np.zeros((state_shape, state_shape)) for _ in range(horizon + 1)]
        self.v_x = [np.zeros((state_shape)) for _ in range(horizon + 1)]
        # Store the Q_uu and Q_u for performing the line search
        self.k = np.zeros((horizon, action_shape))
        self.K = np.zeros((horizon, action_shape, state_shape))


    def fit(self, x0, u_seq, iterations=100, tol=1e-6, on_iteration=None):
        self.mu = 1.0
        self.delta = self.delta_0
        alphas = 1.1**(-np.arange(10)**2)

        us = u_seq.copy()
        k = self.k
        K = self.K

        changed = True
        converged = False
        for iteration in range(iterations):
            accepted = False
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, 
                    L_xx, L_ux, L_uu) = self._forwardRollout(x0, u_seq)
                J_opt = L.sum()
                changed = False
            try:
                k, K = self._backwardPass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu)
                # forward pass
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._J(xs_new, us_new)
                    if J_new < J_opt:
                        # print(np.abs((J_opt - J_new) / J_opt))
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True
                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True
                        self._decrease_mu()
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                warnings.warn(str(e))

            if not accepted:
                self._increase_mu()
                break
            if converged:
                break
            
        self.k = k
        self.K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us
    # The parameters of this function is a nominal state and a nominal action
    # X is the state sequence calulated from the system dynamics
    # U is a squence of actions {u_1, u_2, ..., u_n}

    def _forwardRollout(self, x0, u_seq):
        xs = np.empty((self.horizon+1, self.state_shape))
        F_x = np.empty((self.horizon+1,self.state_shape,self.state_shape))
        F_u = np.empty((self.horizon+1,self.state_shape,self.action_shape))
        L = np.empty(self.horizon+1)
        L_x = np.empty((self.horizon+1, self.state_shape))
        L_u = np.empty((self.horizon+1, self.action_shape))    
        L_xx = np.empty((self.horizon+1, self.state_shape, self.state_shape))
        L_ux = np.empty((self.horizon+1, self.action_shape ,self.state_shape))
        L_uu = np.empty((self.horizon+1, self.action_shape, self.action_shape))
        xs[0] = x0
        for i in range(self.horizon):
            x = xs[i]
            u = u_seq[i]
            xs[i+1] = self.dynamics(x,u)
            F_x[i] = self.f_x(x, u)
            F_u[i] = self.f_u(x, u)
            L[i] = self.running_cost(x, u)
            L_x[i] = self.l_x(x, u)
            L_u[i] = self.l_u(x, u)
            L_xx[i] = self.l_xx(x, u)
            L_ux[i] = self.l_ux(x, u)
            L_uu[i] = self.l_uu(x, u)
        x = xs[-1]
        L[-1] = self.final_cost(x)
        L_x[-1] = self.l_f_x(x)
        L_xx[-1] = self.l_f_xx(x)
        return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, 

    def _backwardPass(self, F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu):
        V_x = L_x[-1]
        V_xx = L_xx[-1]
        k = np.empty_like(self.k)
        K = np.empty_like(self.K)
        for i in range(self.horizon-1,-1,-1):
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i], V_x, V_xx)
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)
        return np.array(k), np.array(K)

    def _Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx):
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        reg = self.mu * np.eye(self.state_shape)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def _control(self, xs, us, k, K, alpha=1):
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()
        for i in range(self.horizon):
            us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])
            xs_new[i+1] = self.dynamics(xs_new[i], us_new[i])
        return xs_new, us_new

    # compute the total cost 
    def _J(self, x_seq, u_seq):
        running_costs = [self.running_cost(x,u) for x,u in zip(x_seq[:-1], u_seq)]
        final_cost = self.final_cost(x_seq[-1])
        j = np.sum(running_costs) + final_cost
        return j

    def _decrease_alpha(self):
        self.alpha = self.alpha * self.gamma

    # return true if A is positive definte else return False
    def _isPD(self, A):
        try:
            np.linalg.cholesky(A)
            return True
        except:
            return False
            
    # Increase the regulrization term according to the proposed schedule
    def _increase_mu(self):
        self.delta = max(self.delta_0, self.delta*self.delta_0)
        self.mu = max(self.mu_min, self.mu*self.delta)

    def _decrease_mu(self):
        self.delta = min(1/self.delta_0, self.delta/self.delta_0)
        self.mu = self.mu * self.delta if self.mu*self.delta > self.mu_min else 0

# Action,State cost
def l(x, u): # pinilize large actions
    return 1 * np.sum(np.square(u))

# final cost
def l_f(x):
    return 1 * (300*np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3]) + 10*np.square(x[0]))

# System Dynamics
def dynamics(x,u):
    return env._state_eq(x, u)


if __name__ == '__main__':
    h = 50
    env = gym.make('CartPoleContinuous-v0').env
    state = env.reset()
    u = np.random.uniform(-1, 1, (h, 1))
    x_seq = [state.copy()]
    # self, dynamics, running_cost, final_cost, state_shape, action_shape, horizon, mu_max
    iLQR = ILQR(dynamics, l, l_f, state.shape[0], 1, h, 1e10)
    thetas = []
    thetas_dot = []
    for i in range(300):
        x,u = iLQR.fit(state, u,iterations=3)
        c = env.render(mode="rgb_array")
        img = Image.fromarray(c)
        # img.save(f'snapshots/{t}.png')
        action = np.clip(u[0], -20, 20)
        obs,_,_,_ = env.step(action)
        state = obs.copy()
        thetas.append(state[2])
        thetas_dot.append(state[3])
        print(f"Time step:{i}, action: {action}")

plt.plot(thetas, thetas_dot)
plt.savefig('phase diagram')
plt.show()
    # print(u_seq)








