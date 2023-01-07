```python
import numpy as np
import matplotlib.pyplot as plt
```

# 1. Known data association


```python
def generate_data(num, R = np.eye(2), t = np.zeros([2,1])):
    data = np.zeros([2,num])
    data[0,:] = range(num)
    data[1,:] = 0.2 * data[0,:] * np.sin(0.5 * data[0,:])
    data = R @ data + t
    return data

def plot_data(data1, data2, data1_label = 'data1', data2_label = 'data2', pt_size1 = 6, pt_size2 = 6):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.plot(data1[0], data1[1], color='blue', marker='o', linestyle=":", label= data1_label, markersize = pt_size1)
    ax.plot(data2[0], data2[1], color='red', marker='o', linestyle=":", label= data2_label, markersize = pt_size2)
    ax.legend()
    return ax
```


```python
# generate data
num = 30
angle = np.pi / 4
R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
T = np.array([[2],[5]])

p = generate_data(num, R, T)
q = generate_data(num)

ax = plot_data(q, p, 'q', 'p')
plt.show()

# rotation and translation from p to q
true_R = R.transpose()
true_T = -T
```


    
![png](icp_example_files/icp_example_3_0.png)
    



```python
# equal weights pn
pn = np.ones(num)

p0 = np.zeros([2,1])
p0[0, 0] = (p[0, :] * pn).sum() / pn.sum()
p0[1, 0] = (p[1, :] * pn).sum() / pn.sum()

q0 = np.zeros([2,1])
q0[0, 0] = (q[0, :] * pn).sum() / pn.sum()
q0[1, 0] = (q[1, :] * pn).sum() / pn.sum()

H = np.zeros([2,2])
for i in range(num):
    b = q[:,i] - q0[:,0]
    a = p[:,i] - p0[:,0]
    H += np.outer(a, b) * pn[i]
    
u, s, vh = np.linalg.svd(H)

cal_R = (u @ vh).transpose()
cal_t = q0 - cal_R @ p0

# now shift p by cal_R and cal_t
cal_p = cal_R @ p + cal_t

ax = plot_data(q, cal_p, 'q', 'cal_p', 8, 4)
plt.show()
```


    
![png](icp_example_files/icp_example_4_0.png)
    


# 2. Unkown data association

### Closest points (Vanilla ICP)


```python
def centralize(p, q):
    # get center
    p0 = np.zeros([2,1])
    p0[0,0] = p[0, :].sum() / p.shape[1]
    p0[1,0] = p[1, :].sum() / p.shape[1]
    
    q0 = np.zeros([2,1])
    q0[0,0] = q[0, :].sum() / q.shape[1]
    q0[1,0] = q[1, :].sum() / q.shape[1]
    
    # move p to center of q
    p = p - p0 + q0
    return p
```


```python
new_p = centralize(p,q)
ax = plot_data(q, new_p, 'q', "p'")
plt.show()
```


    
![png](icp_example_files/icp_example_8_0.png)
    



```python
def get_l2Norm(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_closest_pair(p, q):
    pair = []
    for i in range(p.shape[1]):
        min_norm = np.inf
        correspond_id = -1
        for j in range(q.shape[1]):
            norm = get_l2Norm(p[:,i], q[:,j])
            if norm < min_norm:
                min_norm = norm
                correspond_id = j
        pair.append((i, correspond_id))
    return pair

def draw_correspondencies(p, q, pair, ax):
    for i, j in pair:
        x = [p[0, i], q[0, j]]
        y = [p[1, i], q[1, j]]
        ax.plot(x, y, color = 'grey')
```


```python
ax = plot_data(q, new_p, 'q', "p'")
pair = find_closest_pair(new_p, q)
draw_correspondencies(new_p, q, pair, ax)
plt.show()
```


    
![png](icp_example_files/icp_example_10_0.png)
    



```python
def compute_R_t(p, q, pair, pn):
    p0 = np.zeros([2,1])
    p0[0, 0] = (p[0, :] * pn).sum() / pn.sum()
    p0[1, 0] = (p[1, :] * pn).sum() / pn.sum()

    q0 = np.zeros([2,1])
    q0[0, 0] = (q[0, :] * pn).sum() / pn.sum()
    q0[1, 0] = (q[1, :] * pn).sum() / pn.sum()

    H = np.zeros([2,2])
    for i, j in pair:
        b = q[:,j] - q0[:,0]
        a = p[:,i] - p0[:,0]
        H += np.outer(a, b) * pn[i]

    u, s, vh = np.linalg.svd(H)

    cal_R = (u @ vh).transpose()
    cal_t = q0 - cal_R @ p0

    return cal_R, cal_t
```

## single iteration


```python
# equal weights pn
pn = np.ones(num)

p = centralize(p,q)
pair = find_closest_pair(p, q)

cal_R, cal_t = compute_R_t(p, q, pair, pn)
# shift p to new location
p = cal_R @ p + cal_t

plot_data(q, p, 'q', "p'", 8, 4)
```




    <AxesSubplot: >




    
![png](icp_example_files/icp_example_13_1.png)
    


## with more iteration


```python
# generate data
num = 30
angle = np.pi / 4
R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
T = np.array([[2],[5]])

p = generate_data(num, R, T)
q = generate_data(num)

show_frame = [1, 3, 5]
# 5 is enough for convergence
for i in range(5):
    p = centralize(p,q)
    pair = find_closest_pair(p, q)

    cal_R, cal_t = compute_R_t(p, q, pair, pn)
    # shift p to new location
    p = cal_R @ p + cal_t
    
    if i+1 in show_frame:
        ax = plot_data(q, p, 'q', "p'", 8, 4)
        ax.set_title('iteration: {}'.format(i+1))
        plt.show()

```


    
![png](icp_example_files/icp_example_15_0.png)
    



    
![png](icp_example_files/icp_example_15_1.png)
    



    
![png](icp_example_files/icp_example_15_2.png)
    



```python

```
