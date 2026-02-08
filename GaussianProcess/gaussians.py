import dataclasses
import functools
import numpy as np
import scipy.linalg
import scipy.special
from typing import Callable, Union, Tuple
from tueplots.constants.color import rgb

@dataclasses.dataclass
class Gaussian:
    """Gaussian distribution with mean mu and covariance Sigma"""
    __array_priority__ = 1000  # this is needed to make sure that the @ operator works correctly when the Gaussian is on the right-hand side of the operator
    mu: np.ndarray
    Sigma: np.ndarray

    @functools.cached_property
    def L(self):
        # Cholesky decomposition of the covariance matrix
        return np.linalg.cholesky(self.Sigma)

    @functools.cached_property
    def L_factor(self):
        """Cholesky factor of the covariance matrix"""
        # Cholesky decomposition of the covariance matrix
        return scipy.linalg.cho_factor(self.Sigma, lower=True)[0]


    @functools.cached_property
    def cov_SVD(self):
        """square root of the covariance matrix, via SVD"""
        if np.isscalar(self.mu):
            return np.eye(1), np.sqrt(self.Sigma).reshape(1, 1)
        else:
            Q, S, _ = np.linalg.svd(self.Sigma, full_matrices=True, hermitian=True)
            return Q, np.sqrt(S)

    @functools.cached_property
    def logdet(self):
        """log-determinant of the covariance matrix"""
        _, S = self.cov_SVD
        return 2 * np.sum(np.log(S))
    

    @functools.cached_property
    def precision(self):
        """precision matrix (inverse of Sigma)"""
        Q, S = self.cov_SVD
        return Q @ np.diag(1 / S**2) @ Q.T

    def prec_mult(self, x: np.ndarray):
        """precision matrix multiplication: Sigma^{-1} @ x"""
        Q, S = self.cov_SVD
        return Q @ np.diag(1 / S**2) @ Q.T @ x

    @functools.cached_property
    def mp(self):
        """precision-adjusted mean"""
        return self.prec_mult(self.mu)

    def log_pdf(self, x):
        # log-probability density function of the Gaussian
        d = self.mu.shape[0]
        # x = x[:, None] if x.ndim == 1 else x 
        if x.ndim == 1:
            return (
                -0.5 * (x - self.mu).T @ self.prec_mult(x - self.mu)
                - 0.5 * self.logdet   
                - 0.5 * d * np.log(2 * np.pi)
            )
        elif x.ndim == 2:
            # x is a batch of vectors, shape (N, D)
            N = x.shape[0]
            return (
                -0.5 * np.einsum("ij,jk,ik->i", x - self.mu, self.precision, x - self.mu)
                - 0.5 * self.logdet
                - 0.5 * d * np.log(2 * np.pi)
            )   # Shape (N,)
        else:   
            raise ValueError("x must be a 1D or 2D array")
        

    def pdf(self, x):
        """N(x;mu,Sigma)"""
        return np.exp(self.log_pdf(x))

    def cdf(self, x):
        if np.isscalar(self.mu) or np.size(self.mu) == 1:
            return 0.5 * (
                1 + scipy.special.erf((x - self.mu) / np.sqrt(2 * self.Sigma))
            )
        else:
            raise NotImplementedError("CDF for multivariate Gaussian not implemented")


    def __mul__(self, other, return_log_normalizer: bool = False):
        """Product of two Gaussian PDFs"""
        Sigma = np.linalg.inv(self.precision + other.precision)
        mu = Sigma @ (self.mp + other.mp)
        
        if return_log_normalizer:
            Z = Gaussian(
                mu=other.mu, 
                Sigma=self.Sigma + other.Sigma
            ).log_pdf(self.mu)
            return Gaussian(mu=mu, Sigma=Sigma), Z
        else:
            return Gaussian(mu=mu, Sigma=Sigma)

    def __rmatmul__(self, A: np.ndarray):
        """
        Linear map: y = Ax
        Standard Textbook Definition:
        New Mean: A @ mu
        New Cov:  A @ Sigma @ A.T
        """
        return Gaussian(mu=A @ self.mu, Sigma=A @ self.Sigma @ A.T)

    def __getitem__(self, i):
        """marginals"""
        return Gaussian(
            mu=np.atleast_1d(self.mu[i]), 
            Sigma=np.atleast_2d(self.Sigma[np.ix_(i, i)]) if not np.isscalar(self.Sigma) else self.Sigma
        )

    @functools.singledispatchmethod
    def __add__(self, other: Union[np.ndarray, float]):
        """Affine map: x + constant"""
        other = np.asarray(other)
        return Gaussian(mu=self.mu + other, Sigma=self.Sigma)

    def condition(self, A, y, Lambda):
        # conditioning of a Gaussian RV on a linear observation
        # A: observation matrix
        # y: observation
        # Lambda: observation noise covariance
        Gram = A @ self.Sigma @ A.T + Lambda
        L = scipy.linalg.cho_factor(Gram, lower=True)
        y = np.asarray(y)

        if y.ndim > 1:
            # Case 1: y is a Batch Matrix (N, K)
            # We broadcast prior_pred from (N) to (N, 1) so it subtracts column-wise
            # Add to mean (F). We broadcast self.mu from (F) to (F, 1)
            mu = self.mu[:, None] + self.Sigma @ A.T @ scipy.linalg.cho_solve(L, y - A @ self.mu[:, None])
            
        else:
            # Case 2: y is a Single Vector (N)
            mu = self.mu + self.Sigma @ A.T @ scipy.linalg.cho_solve(L, y - A @ self.mu)
            
        Sigma = self.Sigma - self.Sigma @ A.T @ scipy.linalg.cho_solve(L, A @ self.Sigma)
        return Gaussian(mu=mu, Sigma=Sigma)
    
    @functools.cached_property
    def std(self):
        """standard deviation"""
        if np.isscalar(self.mu):
            return np.sqrt(self.Sigma)
        else:
            return np.sqrt(np.diag(self.Sigma))
        
    def sample(self, n: int = 1, random_state: Union[int, np.random.Generator] = None) -> np.ndarray:
        """Generate samples from the Gaussian distribution
        Parameters        ----------
        n : int
            Number of samples to generate
        random_state : int or np.random.Generator, optional 
            Random seed or random number generator for reproducibility (default: None)  
        Returns
        ------- 
        samples : np.ndarray
            Generated samples of shape (n, d) where d is the dimensionality of the Gaussian distribution    
        """
        if isinstance(random_state, np.random.Generator):
            rng = random_state  
        else:
            rng = np.random.default_rng(random_state)   

        # Alternative:return np.einsum('ij,kj->ki', self.cov_SVD[0], self.cov_SVD[1] * rng.normal(size=(n, self.mu.shape[0]))) + self.mu[:, None] 
        if np.isscalar(self.mu):
            samples = rng.normal(loc=self.mu, scale=np.sqrt(self.Sigma), size=n)
        else:
            samples = rng.multivariate_normal(mean=self.mu, cov=self.Sigma, size=(n, ), method='svd')
        return samples


@Gaussian.__add__.register
def _add_gaussians(self, other: Gaussian):
    """Addition of two Gaussian random variables"""
    return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)


####################################################################################################################################################



@dataclasses.dataclass
class GaussianProcess:
    """ Gaussian Process defined by mean function m and covariance function k """
    # mean function
    m: Callable[[np.ndarray], np.ndarray] 
    # covariance function
    k: Callable[[np.ndarray, np.ndarray], np.ndarray]
    
    def __call__(self, X: np.ndarray) -> Gaussian:
        return Gaussian(mu=self.m(X), Sigma=self.k(X[:, None, :], X[None, :, :])) # Broadcasting for pairwise covariance
    
    def condition(self, X: np.ndarray, y: np.ndarray, Lambda: np.ndarray) -> Gaussian:
        """
        Condition the GP on observations (X, y) with observation noise covariance Lambda
        Parameters
        ----------
        X : np.ndarray
            Input locations of observations (can be referred to as training points). 
            Shape (N, D) or (N,) for 1D inputs. The method will handle both cases.
        y : np.ndarray
            Observations corresponding to input locations X. Shape (N,).    
        Lambda : np.ndarray or float
            Observation noise covariance. Can be a scalar (isotropic noise) or a matrix (anisotropic noise). 
            If scalar, it will be converted to Lambda * I where I is the identity matrix of appropriate size.
        """
        return ConditionalGaussianProcess(self, y, X, Gaussian(mu=np.zeros_like(y), Sigma=Lambda * np.eye(len(y))))
    
    def plot(self, ax, X: np.ndarray, mean_kwargs ={}, std_kwargs ={}, sampled_fun_kwargs ={}, std_lines_kwargs ={}, f_range=(-3, 3), f_resolution=300, 
             num_samples=0, color=rgb.tue_gray, rng: Union[int, np.random.Generator] = None, **kwargs):
        """Plot the GP mean and uncertainty"""
        gp_x = self(X)
        ax.plot(X, gp_x.mu, color=color, **mean_kwargs)
        ff = np.linspace(f_range[0], f_range[1], f_resolution)
        shading = gp_shading(ff[:, None], gp_x.mu[None, :], gp_x.std) 
        ax.imshow(
            shading, 
            extent=(X[0, 0], X[-1, 0], f_range[0], f_range[1]), 
            aspect='auto', origin='lower', **std_kwargs, **kwargs)

        ax.plot(X[:, 0], gp_x.mu + 2 * gp_x.std, **std_lines_kwargs, label="GP 2 Std Dev")
        ax.plot(X[:, 0], gp_x.mu - 2 * gp_x.std, **std_lines_kwargs)

        # ax.fill_between(X[:, 0], gp_x.mu - 2 * gp_x.std, gp_x.mu + 2 * gp_x.std, color=color, **std_kwargs)
        
        if num_samples > 0:
            ax.plot(X[:, 0], gp_x.sample(n=num_samples, random_state=rng).T, **sampled_fun_kwargs)


def gp_shading(ff, mu, std):
    """Compute shading values for GP plot"""
    return np.exp(-0.5 * ((ff - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))





class ConditionalGaussianProcess(GaussianProcess):
    """
    Gaussian Process with conditioning on observations
    Parameters
    ----------  
    prior : GaussianProcess
        Prior Gaussian Process  (object of class GaussianProcess that defines mean and covariance functions, gp.m and gp.k)
    y : np.ndarray
        Observations    
    X : np.ndarray
        Input locations of observations (can be referred to as training points). 
        Shape (N, D) or (N,) for 1D inputs. The class will handle both cases.
    epsilon : Gaussian
        Observation noise model 
        
    """
    
    def __init__(self, prior, y, X, epsilon: Gaussian):
        self.prior = prior
        self.y = np.atleast_1d(y)
        if X.ndim == 1:
            self.X = X[:, None]  # Ensure X is 2D with shape (N, 1)
        else:
            self.X = X
        self.epsilon = epsilon
        # We pass the bound methods directly. No np.vectorize needed.
        super().__init__(self._mean, self._covariance)

    @functools.cached_property
    def predictive_covariance(self):
        # Compute K(X, X) + Sigma_obs
        # We reshape X to use the prior's broadcasting logic
        K_XX = self.prior.k(self.X[:, None, :], self.X[None, :, :])
        return K_XX + self.epsilon.Sigma

    @functools.cached_property
    def predictive_covariance_cholesky(self):
        return scipy.linalg.cho_factor(self.predictive_covariance, lower=True)

    @functools.cached_property
    def representer_weights(self):
        L = self.predictive_covariance_cholesky
        # Solve (K + Sigma)^-1 (y - m(X))
        return scipy.linalg.cho_solve(L, self.y - self.prior.m(self.X))

    def _mean(self, x: np.ndarray) -> np.ndarray:
        '''
        This calculates the posterior mean using the representer theorem:
        m_post(x) = m_prior(x) + k_prior(x, X) @ weights where weights = (K_XX + Sigma)^-1 (y - m(X))
        Parameters
        ---------
        x : np.ndarray
            Input locations where we want to evaluate the posterior mean (can be referred to as test points). 
            Shape (M, D) or (M,) for 1D inputs. The method will handle both cases.

        '''
        x = np.asarray(x)
        # Predictive Mean: m(x) + k(x, X) @ weights
        # We broadcast x against self.X correctly
        # x is (N, D), self.X is (M, D)
        # prior.k needs inputs shaped to broadcast -> (N, 1, D) vs (1, M, D)
        K_xX = self.prior.k(x[..., None, :], self.X[None, :, :])
        return self.prior.m(x) + K_xX @ self.representer_weights

    def _covariance(self, x1, x2):
        # This calculates the posterior covariance using the Schur Complement:
        # K_post = K_prior(x, x) - K_prior(x, X) @ (K_XX + Sigma)^-1 @ K_prior(X, x)
        
        # 1. Prior Covariance K(x, x)
        # x is (M, 1, D), x is (1, M, D) -> Result is (M, M)
        K_xx = self.prior.k(x1, x2) # Shape (M, M)
        
        # 2. Cross Covariance K(x, X_train)
        # x is (M, 1, D). We reshape self.X to (1, N, D)
        K_xX = self.prior.k(x1, self.X[None, :, :]) # Shape (M, N) where N is number of training points
        
        # 3. Cross Covariance K(X_train, x)
        # We reshape self.X to (N, 1, D). x is (1, M, D)
        K_Xx = self.prior.k(self.X[:, None, :], x2) # Shape (N, M)

        # 4. Compute Correction Term
        L = self.predictive_covariance_cholesky
        # Solves L L.T x = K_Xx  ->  (K + Sigma)^-1 K_Xx
        return K_xx - K_xX @ scipy.linalg.cho_solve(L, K_Xx)



class ParametricGaussianProcess(GaussianProcess):
    """Parametric special case of a Gaussian Process"""
    def __init__(self, phi: Callable, prior: Gaussian):
        self.phi = phi
        self.prior = prior
        super().__init__(self._mean, self._covariance)

    def _mean(self, x):
        return self.phi(x) @ self.prior.mu

    def _covariance(self, x, y):
        return self.phi(x) @ self.prior.Sigma @ self.phi(y).T









    # def __init__(self, prior, y, X, epsilon: Gaussian):
    #     self.prior = prior
    #     self.y = np.atleast_1d(y) # ensure y is 1D (n_samples,)
    #     self.X = np.atleast_2d(X) # ensure X is 2D (n_samples, n_features)
    #     self.epsilon = epsilon
    #     super().__init__(self._mean, self._covariance)

    # @functools.cached_property
    # def predictive_covariance(self):
    #     K_XX = self.prior.k(self.X[:, None, :], self.X[None, :, :])
    #     return K_XX + self.epsilon.Sigma
    
    # @functools.cached_property
    # def predictive_covariance_cholesky(self):
    #     return scipy.linalg.cho_factor(self.predictive_covariance, lower=True)
    
    # @functools.cached_property
    # def representer_weights(self):
    #     L = self.predictive_covariance_cholesky
    #     return scipy.linalg.cho_solve(L, self.y - self.prior(self.X).mu - self.epsilon.mu)

    # def _mean(self, x: np.ndarray) -> np.ndarray:
    #     x = np.asarray(x)
    #     return self.prior(x).mu + self.prior.k(x[..., None, :], self.X[None, :, :]) @ self.representer_weights

    # @functools.partial(np.vectorize, signature='(d),(d)->()', excluded={0})
    # def _covariance(self, a, b):
    #     K_ab = self.prior.k(a, b)
    #     K_aX = self.prior.k(a, self.X)
    #     K_Xb = self.prior.k(self.X, b)
    #     L = self.predictive_covariance_cholesky
    #     return K_ab - K_aX @ scipy.linalg.cho_solve(L, K_Xb)
    
    # def _m_proj(self, x: np.ndarray, projection, projection_mean) -> np.ndarray:
    #     """Mean function after projection"""
    #     x = np.asarray(x)
    #     if projection_mean is None:
    #         projection_mean = self.prior.mu 
    #     return projection @ (self.prior(self.X).mu + self.prior.k(x[..., None, :], self.X[None, :, :]) @ self.representer_weights) + projection_mean(x)









    # def plot(self, ax, X: np.ndarray, mean_kwargs ={}, std_kwargs ={}, y_range=(-3, 3), yres=300, 
    #          num_samples=0, color=rgb.tue_gray, rng: Union[int, np.random.Generator] = None, **kwargs):
    #     """Plot the conditional GP mean and uncertainty"""
    #     gp_x = self(X)
    #     ax.plot(X, gp_x.mu, color=color, **mean_kwargs)
    #     yy = np.linspace(y_range[0], y_range[1], yres)
    #     ax.imshow(
    #         np.exp(gp_x.log_pdf(yy[:, None], gp_x.mu[None, :], gp_x.std)).T,
    #         extent=(X[0, 0], X[-1, 0], y_range[0], y_range[1]), **std_kwargs, 
    #         aspect='auto',
    #         origin='lower',
    #         cmap='Greys',
    #         alpha=0.5,
    #         **kwargs
    #     )
    #     ax.plot(X[:, 0], gp_x.mu + 2 * gp_x.std, color=color, linestyle='--', lw=0.25)
    #     ax.plot(X[:, 0], gp_x.mu - 2 * gp_x.std, color=color, linestyle='--', lw=0.25)
    #     # ax.fill_between(
    #     #     X[:, 0], 
    #     #     gp_x.mu - 2 * gp_x.std, 
    #     #     gp_x.mu + 2 * gp_x.std, 
    #     #     color=color, 
    #     #     **std_kwargs
    #     # )
    #     if num_samples > 0:
    #         ax.plot(X[:, 0], gp_x.sample(n=num_samples, random_state=rng).T, color=color, alpha=0.2, lw=0.75)






#######################################################################################
def poly_features(x, degree):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else:
        x = x.reshape(-1, 1)
    d = degree + 1
    return x ** np.arange(d).reshape(1, -1) / np.exp(scipy.special.gammaln(np.arange(d).reshape(1, -1) + 1)) / np.sqrt(d) 

def gaussian_features(x, num_features=2, sigma=1.0):
    """Feature map for a Gaussian basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return np.exp(
        -(((x - np.linspace(-8, 8, num_features)) / sigma) ** 2)
    ) / np.sqrt(num_features)


def relu_features(x, num_features=2):
    """Feature map for a ReLU basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return np.maximum(
        0,
        (np.sign(np.arange(0, num_features) % 2 - 0.5))
        * (x - np.linspace(-8, 8, num_features)),
    ) / np.sqrt(num_features)

def one_sided_relu(x, num_features=2):
    """Feature map for a ReLU basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return np.maximum(
        0,(x - np.linspace(-8, 8, num_features)),
    ) / np.sqrt(num_features)

def cosine_features(x, num_features=2, ell=1.0):
    """Feature map for a cosine basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)  
    return np.cos(np.pi * x / np.arange(1, num_features) / ell) / np.sqrt(
        num_features
    )


def trig_features(x, num_features=2, ell=1.0):
    """Feature map for a combination of cosine and sine basis functions."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return np.hstack(
        [
            np.cos(np.pi * x / np.arange(1, np.floor(num_features / 2.0)) / ell),
            np.sin(np.pi * x / np.arange(1, np.ceil(num_features / 2.0)) / ell),
        ]
    ) / np.sqrt(num_features)


def sigmoid_features(x, num_features=2, ell=1.0):
    """Feature map for a sigmoid basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return (
        1
        / (1 + np.exp(-(x - np.linspace(-8, 8, num_features)) / ell))
        / np.sqrt(num_features)
    )


def step_features(x, num_features=2):
    """Feature map for a step basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return np.sign(x - np.linspace(-8, 8, num_features)) / np.sqrt(num_features)


def switch_features(x, num_features=2):
    """Feature map for a switch basis function."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else: 
        x = x.reshape(-1, 1)
    # output shape: (n_samples, order)
    return (x > np.linspace(-8, 8, num_features)) / np.sqrt(num_features)

#######################################################################################
