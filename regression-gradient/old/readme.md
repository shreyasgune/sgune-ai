# Regression Gradient Descent

## Setup
`python -m venv venv`

`source venv/bin/activate`
>On Windows use `venv\Scripts\activate` 

pip install -r requirements.txt 

### Test Graphing
```python

import torch
from torchviz import make_dot

v = torch.tensor(1.0, requires_grad=True)
make_dot(v)

```

and output would be : `<graphviz.graphs.Digraph object at 0x0000027017EEE6E0>` 

--------------
## Linear Regression Model 

Gradient descent is an iterative technique commonly used in machine learning and deep learning to find the best possible set of parameters / coefficients for a given model, data points, and loss function, starting from an initial, and usually random, guess.

## The model 

`y = b + wx + e` 
where 
 - b = bias, which tells the expected average value y when x = 0.
 - w = weight, which tells how much y increases when x increase by 1.
 - e = _epsilon_ or error. Its the noise we can't get rid of. 

For our example, lets use b = 1 and w = 2.
Gaussian noise added using `randn()` 


- Generated synthetic data from a linear model (y = b + w*x + noise).

- Split the data into training and validation sets.

- Initialized random b and w.

- Calculated predictions and MSE loss.

- Created a grid of possible b and w values.

- Computed the full MSE loss surface across that grid using training data.

- Calculated gradients at the current point (b, w).


Initial Loss:
```
Initial loss: 2.7422

when 
random seed is set to 42
b=[0.49671415] and w=[-0.1382643]
```


Mesh Grid Matrix:
```
[[-2.   -1.94 -1.88 ...  3.88  3.94  4.  ]
 [-2.   -1.94 -1.88 ...  3.88  3.94  4.  ]
 [-2.   -1.94 -1.88 ...  3.88  3.94  4.  ]
 ...
 [-2.   -1.94 -1.88 ...  3.88  3.94  4.  ]
 [-2.   -1.94 -1.88 ...  3.88  3.94  4.  ]
 [-2.   -1.94 -1.88 ...  3.88  3.94  4.  ]]
```



- Training Data plot

![](regression-gradient\images\training_data.png)

- Validation Data plot
  
![](regression-gradient\images\validation.png)


##

