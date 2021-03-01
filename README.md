<h1 align="center"> 
  Surrogate Models
</h1>


A collection of surrogate models (wrapper classes) for sequence model based optimization techniques used in Hyperactive and Gradient-Free-Optimizers.


<br>

## Bayesian Optimization Surrogate Models

<details>
<summary><b> GPy </b></summary>

```python
import GPy
import numpy as np


class GPySurrogateModel:
    def __init__(self):
        self.kernel = GPy.kern.RBF(input_dim=1)

    def fit(self, X, y):
        self.m = GPy.models.GPRegression(X, y, self.kernel)
        self.m.optimize(messages=False)

    def predict(self, X, return_std=False):
        mean, std = self.m.predict(X)

        if return_std:
            return mean, std
        return mean
```

</details>



<details>
<summary><b> GPflow </b></summary>

```python
import gpflow
import numpy as np


class GPflowSurrogateModel:
    def __init__(self):
        self.kernel = gpflow.kernels.Matern52()

    def fit(self, X, y):
        X = X.astype(np.float64)

        self.m = gpflow.models.GPR(data=(X, y), kernel=self.kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            self.m.training_loss, self.m.trainable_variables, options=dict(maxiter=100)
        )

    def predict(self, X, return_std=False):
        X = X.astype(np.float64)
        mean, std = self.m.predict_f(X)

        mean = np.array(mean)
        std = np.array(std)

        if return_std:
            return mean, std
        return mean
```

</details>