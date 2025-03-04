# PPO Learning

> The PPO algorithm is a policy optimisation algorithm used to train policies in reinforcement learning.The PPO algorithm updates the policy by minimising the KL dispersion of the policy, thus avoiding drastic changes in the policy.The PPO algorithm updates the policy by maximising the objective function, which consists of the payoffs of the current policy and an estimate of its dominance.The PPO algorithm avoids drastic changes in the policy by truncating the objective function, thusimprove the stability of the algorithm.



> This project will use the PPO algorithm to train a policy to be able to reach a target state in a given environment.

This is my python environment:

```python
Python 3.9.20
conda create -n ppo python=3.9
conda activate ppo
pip install -r requirements.txt
```


![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*AX2vLguiKvxn-YIntIx18w.png)

```python 

1. Input: 策略参数，初始值函数参数  
2. for k=0,1,2,.. do:
3.     在env中通过运行策略函数Π_k采集轨迹D_k
4.     计算每次奖励R_t
5.     根据当前的价值函数V_k，计算优势估计值A_t
6.     通过最大化 PPO-Clip 目标来更新政策：通常是通过Adam的随机梯度上升算法。
7.     通过均方误差回归拟合值函数:通常是通过某种梯度下降算法。
8. end for

```

# 整体流程


```mermaid
    flowchart TD
        INIT["初始化PPO类"]
        HYPERPARAMS["初始化超参数"]
        CREATE_NETS["创建策略网络和值函数网络"]
        COLLECT_DATA["收集数据"]
        CALC_RTGS["计算奖励到目标"]
        UPDATE_MODEL["更新模型"]
        END(("END"))
        INIT --> HYPERPARAMS
        HYPERPARAMS --> CREATE_NETS
        CREATE_NETS --> COLLECT_DATA
        COLLECT_DATA --> CALC_RTGS
        CALC_RTGS --> UPDATE_MODEL
        UPDATE_MODEL --> COLLECT_DATA
        UPDATE_MODEL --> END
```


## 代码实现

1. PPO类 - rollout方法

```mermaid
flowchart TD
    start["开始"]
    reset["重置环境，获取初始观测"]
    get_action["调用get_action方法获取动作和动作的对数概率"]
    step["从环境中执行动作，获取新的观测、奖励和是否结束"]
    append_obs["将当前观测添加到batch_obs中"]
    append_action["将动作和动作的对数概率添加到batch_acts和batch_log_probs中"]
    append_ep_rews["将奖励添加到ep_rews中"]
    check_done["检查episode是否结束"]
    compute_rtgs["调用compute_rtgs方法计算奖励的折扣回报"]
    return["返回batch_obs、batch_acts、batch_log_probs、batch_rtgs和batch_lens"]
    start --> reset --> get_action --> step --> append_obs --> append_action --> append_ep_rews --> check_done
    check_done -->|是| compute_rtgs --> return
    check_done -->|否| get_action
    compute_rtgs --> return
```



📕 参考链接：https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a