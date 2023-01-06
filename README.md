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
    

