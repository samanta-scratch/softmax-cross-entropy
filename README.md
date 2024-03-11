# How does "softmax_cross_entropy_with_logits" work ?
## Maths behind:
### Step - 01:
Calculate softmax of logits using equation

#### f(s) = e^s/∑e^s

Here, s is logit

### Step - 02:

Then Calculate Cross Entropy Loss:

#### CE = - ∑ t * log(f(s)) [t is label]
## Step by Step Calculation:
### Step - 01:

f(4.0) = e^4.0 / (e^4.0 + e^2.0 + e^1.0) = 0.8437947

f(0) = e^4.0 / (e^0 + e^5.0 + e^1.0) = 0.00657326

f(2.0) = e^4.0 / (e^4.0 + e^2.0 + e^1.0) = 0.11419519

f(5.0) = e^4.0 / (e^4.0 + e^2.0 + e^1.0) = 0.9755587

f(1.0) = e^4.0 / (e^4.0 + e^2.0 + e^1.0) = 0.04201007

f(1.0) = e^4.0 / (e^4.0 + e^2.0 + e^1.0) = 0.01786798

### Step - 02:

CE = - ∑ t * log(f(s)) [t is label]

= - (1 * log(0.8437947) + 0 * log(0.11419519) + 0 * log(0.04201007) ) , - (0 * log(0.00657326) + 5 * log(0.9755587) + 1 * log(0.01786798) )

= 0.16984606, 0.82474496

## Code Snippet
### Step - 01:
```python
logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
logits_soft = tf.nn.softmax(logits)
logits_soft
```
### Output:

<tf.Tensor: shape=(2, 3), dtype=float32, numpy=

array([[0.8437947 , 0.11419519, 0.04201007],

[0.00657326, 0.9755587 , 0.01786798]], dtype=float32)>
### Step - 02:
```python
ce = np.sum(-(labels * np.log(logits_soft)), axis=1)
ce
```
### Output:

array([0.16984606, 0.82474496])

### Entire syntax:
```python
softmax_cross_entropy_with_logits(labels=labels, logits=logits)
```
### Output:

<tf.Tensor: shape=(2,), dtype=float32,

numpy=array([0.16984604, 0.82474494], dtype=float32)>
