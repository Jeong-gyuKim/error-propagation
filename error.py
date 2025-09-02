import math
import numpy as np
from functools import wraps 
import matplotlib.pyplot as plt

def _try_except_wrapper(self, other, func):
    try:
        return func(self, other)
    except AttributeError:
        return func(Error(self), Error(other))
    except Exception as e:
        raise e

def ndarray_safe(func):
    @wraps(func)
    def ndarray_safe_wrapper(err, target, *args, **kwargs):
        if isinstance(target, np.ndarray) or isinstance(target, list) or isinstance(target, tuple):
            li = [func(err, item, *args, **kwargs) for item in target]
            return np.asarray(li)
        return func(err, target, *args, **kwargs)

    return ndarray_safe_wrapper

class Error:
    """Object to store value and error.
    The error should be the standard deviation of the value."""
    def __init__(self, avg, std = 0.0):
        """
        Args:
            avg: Value
            error: Standard deviation of value
        """
        try:
          float(avg)
        except ValueError:
          raise ValueError(f"Value {type(avg), avg} could not convert to float")
        try:
          float(std)
        except ValueError:
          raise ValueError(f"Error {type(std), std} could not convert to float")
        self.avg = float(avg)
        self.std = abs(float(std))

    def __str__(self):
        if np.isnan(self.avg) or np.isnan(self.std):
            return "NaN"
        if self.std == 0:
            return f"{self.avg} ± {self.std}"
        n = math.floor(np.log10(self.std))
        return f"{round(self.avg*10**-n)/10**-n} ± {round(self.std*10**-n)/10**-n}"
    def __repr__(self):
        return self.__str__()
    def __eq__(self,other):
        other = Error(other)
        return self.avg == other.avg
    def __ne__(self,other):
        other = Error(other)
        return self.avg != other.avg
    def __lt__(self,other):
        other = Error(other)
        return self.avg < other.avg
    def __le__(self,other):
        other = Error(other)
        return self.avg <= other.avg
    def __gt__(self,other):
        other = Error(other)
        return self.avg > other.avg
    def __ge__(self,other):
        other = Error(other)
        return self.avg >= other.avg

    @ndarray_safe
    def __add__(self, other):
        return _try_except_wrapper(self, other, Error.add)
    def __radd__(self, other):
        return self.__add__(other)
    def __iadd__(self, other):
        return self.__add__(other)

    @ndarray_safe
    def __sub__(self, other):
        return _try_except_wrapper(self, other, Error.sub)
    def __rsub__(self, other):
        return -self.__sub__(other)
    def __isub__(self, other):
        return self.__sub__(other)

    @ndarray_safe
    def __mul__(self, other):
        return _try_except_wrapper(self, other, Error.mul)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __imul__(self, other):
        return self.__mul__(other)

    @ndarray_safe
    def __truediv__(self, other):
        return _try_except_wrapper(self, other, Error.div)
    @ndarray_safe
    def __rtruediv__(self, other):
        return _try_except_wrapper(other, self, Error.div)
    def __itruediv__(self, other):
        return self.__truediv__(other)
    
    @ndarray_safe
    def __pow__(self, other):
        return self.pow(other)
    def __ipow__(self, other):
        return self.__pow__(other)
    @ndarray_safe
    def __rpow__(self, other):
        other = Error(other)
        return other.pow(self)

    def __neg__(self):
        return Error(-self.avg, self.std)
    def __pos__(self):
        return self
    def __abs__(self):
        return Error(abs(self.avg), self.std)
    
    def __float__(self):
        return float(self.avg)
    def __int__(self):
        return int(self.avg)
    def __contains__(self, item):
        return abs(self.avg - item) <= self.std

    def add(self, other):
        if not isinstance(other, Error):
            other = Error(other)
        return Error(self.avg + other.avg, 
                     np.sqrt(self.std**2 + other.std**2))
    def sub(self, other):
        if not isinstance(other, Error):
            other = Error(other)
        return Error(self.avg - other.avg, 
                     np.sqrt(self.std**2 + other.std**2))
    def mul(self, other):
        if not isinstance(other, Error):
            other = Error(other)
        if self.avg == 0 or other.avg == 0:
            return Error(0, 0)
        else:
            return Error(self.avg * other.avg, 
                         self.avg * other.avg * np.sqrt(((self.std / self.avg)**2 
                                                         + (other.std / other.avg)**2)))
    def div(self, other):
        if not isinstance(other, Error):
            other = Error(other)
        if other.avg == 0:
            raise ValueError("Cannot divide by zero")
        if self.avg == 0:
            return Error(0, 0)
        return Error(self.avg / other.avg, 
                     self.avg / other.avg * np.sqrt(((self.std / self.avg)**2 
                                                     + (other.std / other.avg)**2)))
    def pow(self, other):
        if not isinstance(other, Error):
            other = Error(other)
        if self.avg == 0:
            return Error(0, (other.avg * self.avg**(other.avg - 1) * self.std))
        return Error(self.avg**other.avg, 
                     np.sqrt((other.avg * self.avg**(other.avg - 1) * self.std)**2 
                             + (abs(self.avg)**other.avg * np.log(abs(self.avg)) * other.std)**2)
                     )
    def sqrt(self):
        return self ** 0.5
    def exp(self):
        return Error(np.exp(self.avg), np.exp(self.avg) * self.std)
    def log(self):
        return Error(np.log(self.avg), self.std / self.avg)
    def sin(self):
        return Error(np.sin(self.avg), np.cos(self.avg) * self.std)
    def cos(self):
        return Error(np.cos(self.avg), np.sin(self.avg) * self.std)
    def string(self, sigma=1):
        sigma = abs(sigma)
        if self.std*sigma == 0:
            return f"{self.avg} ± {self.std*sigma}"
        n = math.floor(np.log10(self.std*sigma))
        return f"{round(self.avg*10**-n)/10**-n} ± {round(self.std*sigma*10**-n)/10**-n}"


def arrays_to_error(values, errors):
    return np.array(
        list(
            map(
                lambda li: Error(
                    li[0], li[1]
                ),
                zip(values, errors)
            )
        )
    )
    
def LinearRegression(x,y):
    x, y = np.array(x), np.array(y)
    a = sum((x-np.mean(x))*(y-np.mean(y)))/sum((x-np.mean(x))**2)
    b = np.mean(y) - a * np.mean(x)
    return a, b


def inverse_matrix(matrix):
    # 행렬 차원 확인
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("행렬은 정방행렬이어야 합니다.")
    identity = np.eye(n)
    augmented_matrix = np.hstack((matrix, identity))
    
    # 가우스-조르단 소거법
    for i in range(n):
        pivot = float(augmented_matrix[i, i])
        augmented_matrix[i] = augmented_matrix[i] / pivot
        for j in range(n):
            if i != j:
                row_factor = float(augmented_matrix[j, i])
                augmented_matrix[j] = augmented_matrix[j] - row_factor * augmented_matrix[i]
    inverse = augmented_matrix[:, n:]
    
    return inverse

def polyfit(x,y,degree):
    X = np.vander(x,degree+1)*Error(1)
    coeffs = inverse_matrix(X.T @ X) @ X.T @ y
    return coeffs

def figure(facecolor="#F0F0F0",*args,**kwargs):
    plt.figure(facecolor=facecolor, *args,**kwargs)

def errorscatter(x,y, sigma=1., alpha=.75, fmt="None", capsize=3, capthick=1, label='Experiment value',*args,**kwargs):
    sigma = abs(sigma)
    x,y = x*Error(1),y*Error(1)
    xval = np.array([item.avg for item in x])
    xerr = np.array([sigma*item.std for item in x])
    yval = np.array([item.avg for item in y])
    yerr = np.array([sigma*item.std for item in y])

    plt.errorbar(xval,yval,yerr,xerr, alpha=alpha,fmt=fmt,capsize=capsize,capthick=capthick,label=label,*args,**kwargs)

def errorplot(x, y, x_i=None, x_f=None, sigma=1., label=None, *args, **kwargs):
    sigma = abs(sigma)
    a, b = LinearRegression(x,y)
    x_i = x_i if x_i is not None else min(x)
    x_f = x_f if x_f is not None else max(x)
    x = np.array([x_i,x_f])
    y = np.array(a * x + b)
    
    plt.plot(x.astype(float), y.astype(float), label=label if label is not None else f'linear regression\ny=({a.string(sigma)})*x+({b.string(sigma)})', *args, **kwargs)
    
    
def errorfill(x, y, x_i=None, x_f=None, sigma=1., alpha=0.2, label=None, *args, **kwargs):
    sigma = abs(sigma)
    a, b = LinearRegression(x,y)
    x_i = x_i if x_i is not None else min(x)
    x_f = x_f if x_f is not None else max(x)
    x = np.array([x_i,x_f])
    y_upper = np.array((a.avg + sigma*a.std) * x + (b.avg + sigma*b.std)).astype(float)
    y_lower = np.array((a.avg - sigma*a.std) * x + (b.avg - sigma*b.std)).astype(float)
    
    plt.fill_between(x.astype(float), y_lower, y_upper, alpha=alpha, label=label if label is not None else f'linear regression\ny=({a.string(sigma)})*x+({b.string(sigma)})', *args, **kwargs)
    
def smoothGaussian(arr,degree=5):
    arr = np.concatenate([arr[0]*np.ones(degree-1), arr, arr[-1]*np.ones(degree)], axis=0)
    window=degree*2-1  
    weight=np.ones(window)
    weightGauss=[]  
    for i in range(window):  
        i=i-degree+1  
        frac=i/float(window)  
        gauss=1/(np.exp((4*(frac))**2))  
        weightGauss.append(gauss)  
    weight=np.array(weightGauss)*weight  
    smoothed=np.zeros(len(arr)-window)  
    for i in range(len(smoothed)):  
        smoothed[i]=sum(np.array(arr[i:i+window])*weight)/sum(weight)  
    return smoothed

def Trapezoidal(x, y, init_index, final_index):
    sol = (x[final_index] - x[init_index]) * (y[init_index] + y[final_index]) / 2
    return sol

def integration(x, y, init_index=None, final_index=None, function=Trapezoidal):
    init_index = 0 if init_index is None else init_index
    final_index = len(x) if final_index is None else final_index
    sol = 0
    for i in np.arange(init_index,final_index-1):
        sol += function(x, y, init_index+i, init_index+i+1)
    return sol

#constant def

pi = np.pi

#from modern phy.
#Elementary charge
e = 1.6022e-19 #C
#Speed of light in vacuum
c = 2.9979e8 #m/s
#Permeability of vacuum (magnetic constant) 
mu_0 = 4*pi*1e-7 #N/A^2
#Permittivity of vacuum (electric constant) 
epsilon_0 = 8.8542e-12 #F/m
#Gravitation constant 
G = 6.6738e-11 #N m^2/kg^2
#Planck constant
h = 6.6261e-34 #J s
hbar = h/(2*pi)
#Avogadro constant 
NA = 6.0221e23 #1/mol
#Boltzmann constant 
k = 1.3807e-23 #J/K
#Stefan-Boltzmann constant 
sigma = 5.6704e-8 #W/m^2 K^4
#Atomic mass unit 
u = 1.66053886e-27 #kg

#mass [kg]
m_e = 9.1094e-31
m_muon = 1.8835e-28
m_Proton = 1.6726e-27
m_Neutron = 1.6749e-27
m_Deuteron = 3.3436e-27
m_alpha = 6.6447e-27

