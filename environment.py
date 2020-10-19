def get_threshold(env, constraint='velocity'):

    if constraint == 'circle':
        return 50
    else:
        # Calculated using 50% of required speed of unconstrained PPO agent
        thresholds = {'Ant-v3': 103.115,
                      'HalfCheetah-v3': 151.989,
                      'Hopper-v3': 82.748,
                      'Humanoid-v3': 20.140,
                      'Swimmer-v3': 24.516,
                      'Walker2d-v3': 81.886}


        return thresholds[env]



