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

- Training Data plot

![training](regression-gradient\images\training_data.png)

- Validation Data plot
  
![validation](regression-gradient\images\validation.png)


