# Linear Regression

## Cost function:

$$
J({w,b})=\frac{1}{2m} \sum\limits_{i=0}^{m-1} \left[ f_{w,b}(x^{(i)}) — y^{(i)} \right]²
$$

### Where:

$$
f_{w,b}(x^{(i)}) = wx^{(i)} + b
$$

This formula represents the linear regression model, where:

- *w* is the weight (or slope)
- *b* is the bias (or y-intercept)
- *x(i)* is the i-th input feature
- *y(i)* is the i-th target value
- *m* is the number of training examples

The goal of linear regression is to find the optimal values for *w* and *b* that minimize the cost function *J(w,b)*.

## Gradient Descent:

Gradient descent is an iterative optimization algorithm used to minimize the cost function. Here are the key equations:

### Update Rules:

$$
w = w - \alpha \frac{\partial J(w,b)}{\partial w}
$$

$$
b = b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

Where α is the learning rate.

### Partial Derivatives:

$$
\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum\limits_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}
$$

$$
\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum\limits_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})
$$

These equations are used iteratively to update w and b until the cost function converges to a minimum.

### Simultaneous Update:

It's important to update w and b simultaneously:

$$
\begin{align*}
temp\_w &= w - \alpha \frac{1}{m} \sum\limits_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}\\
temp\_b &= b - \alpha \frac{1}{m} \sum\limits_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})\\
w &= temp\_w \\
b &= temp\_b
\end{align*}
$$

This process is repeated for a specified number of iterations or until the change in the cost function falls below a certain threshold.

## Normalize the Dataset using min max

Min-max normalization is a common technique used to scale features to a fixed range, typically between 0 and 1. This process helps to standardize the range of independent variables or features of data. The formula for min-max normalization is:

$$
X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}
$$

Similarly, for the target variable Y:

$$
Y_{normalized} = \frac{Y - Y_{min}}{Y_{max} - Y_{min}}
$$

## De-normalize w and b

After normalizing the dataset and training the linear regression model, it's often necessary to de-normalize the learned parameters (w and b) to interpret them in the original scale of the data. This process involves reversing the min-max normalization applied earlier. Here's how to de-normalize w and b:

$$
w_{denormalized} = w * \frac{Y_{max} - Y_{min}}{X_{max} - X_{min}}
$$

$$
b_{denormalized} = b * (Y_{max} - Y_{min}) + Y_{min} - w_{denormalized} * X_{min}
$$
