Implementation for First Order Constraint Optimization in Policy Space (FOCOPS).

Link to paper https://arxiv.org/abs/2002.06506

### Requirements
[pytorch >= 1.3.1](https://pytorch.org/) <br>
[gym >= 0.15.3](https://github.com/openai/gym) <br>
[mujoco-py >= 1.50.1.0](https://github.com/openai/mujoco-py) <br>
For the circle experiments, please also install circle environments at
https://github.com/ymzhang01/mujoco-circle.

### Implementation
Example: Humanoid task in the robots with speed 
limits experiments (using the default parameters)
```
python focops_main.py --env-id='Humanoid-v3' --constraint='velocity'
```



