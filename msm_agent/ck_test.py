# Started from the CK test code on github from Yuqing Zheng (syqzheng@gmail.com) Jun. 24th, 2016

import numpy as np
import matplotlib.pyplot as plt
import sys
from msmbuilder.msm.validation import BootStrapMarkovStateModel

def remaining_probability_from_model(num_states, num_steps, total_states, tprob, pop_sort):
    """
    Calculate the remaining probability for top populated states after certain
    number of Markov steps from the Markov State Model.
    """
    prob_model = []
    for i in range(1,num_states+1):
        prob = np.zeros(total_states)
        prob[pop_sort[-i]] = 1.0
        prob_state = []
        for j in range(1, num_steps+1):
            prob = np.dot(prob, tprob)
            prob_state.append(prob[pop_sort[-i]])
        prob_model.append(prob_state)
    return np.array(prob_model) # [num_states, num_steps]


def remaining_probability_from_data(num_states, num_steps, markov_step, pop_sort, data):
    """
    Calculate the remaining probability for top populated states after certain
    number of Markov steps from trajectories.
    """
    length = len(data[0])
    state_flag = []
    prob_data = []
    for i in range(1,num_states+1):
        print('Calculating state', i)
        lag_flag = []
        prob_state = []
        for k in range(1,num_steps+1): # to get n x Markov
            markov_step_new = markov_step*k
            state = pop_sort[-i]
            flag = []
            for traj in data:
                for j in range(length - markov_step_new):
                    if traj[j+markov_step_new] != -1 and traj[j] == state:
                        if traj[j+markov_step_new] == state:
                            flag.append(1)
                        else:
                            flag.append(0)
            lag_flag.append(flag)
            prob_state.append(float(sum(flag))/len(flag))
        prob_data.append(prob_state)
        state_flag.append(lag_flag)
    return state_flag, prob_data


def block_average(x, block_size):
    """
    Calculate uncertainty of remaining probability using block averaging.
    """
    block_means = []
    for i in range(0, len(x)/block_size):
        block_means.append(np.mean(x[block_size*i:block_size*(i+1)]))         
    sigma = np.std(block_means)/np.sqrt(len(x)/block_size-1)
    return sigma

def get_data_standard_error(num_states, num_steps, state_flag, block_sizes):
    """
    Calculate uncertainty of remaining probability for each state at each Markov step.
    """
    standard_error = []
    for i in range(0, num_states):
        error = []
        for j in range(0, num_steps):
            error.append(block_average(state_flag[i][j], block_sizes[i]))
        standard_error.append(error)
    return np.array(standard_error) # [num_states, num_steps]

def get_model_standard_error(num_states, num_steps, clustred_trajs, mdl, n_samples: int = 100):
    """
    Calculate uncertainty of remaining probability from bootstrapped Markov State Model.
    total_states, tprob, pop_sort
    """
    bmsm = BootStrapMarkovStateModel(n_samples=n_samples, 
                                     msm_args={"lag_time":mdl.lag_time_,
                                               "n_timescales":mdl.n_timescales_,
                                               "reversible_type":mdl.reversible_type_,
                                               "ergodic_cutoff":mdl.ergodic_cutoff_,
                                               }, save_all_models=True)
    bmsm.fit(clustred_trajs)
    b_remaining_p = []
    for msm in bmsm.all_models_:
        pop = sorted(range(len(msm.populations_)), key=lambda k: msm.populations_[k])
        remain_p = remaining_probability_from_model(num_states, num_steps, len(mdl.state_labels_), msm.transmat_, pop) # [num_states, num_steps]
        b_remaining_p.append(remain_p) # [n_samples, num_states, num_steps]
    return np.std(np.array(b_remaining_p),axis=0)  # [num_states, num_steps]

def plot_ck_test(num_states, num_steps, prob_data, standard_error, prob_model, outpath):
    """
    Plot the CK test results for each state and save.
    """
    lag = range(1, num_steps+1)
    plt.figure(figsize=(8,2.5*num_states))
    fs = 10
    
    position = []
    for i in range(1, num_states+1):
        position.append(int(str(num_states)+'1'+str(i)))
    cl1, cl2 = 'blue', 'magenta'
        
    for i in range(0, num_states):
        plt.subplot(position[i])
        plt.plot(lag, prob_data[i], c=cl1, label='From Data', lw=2.5, marker='o', markeredgecolor=cl1, markersize=8)
        plt.errorbar(lag, prob_data[i], c=cl1,  yerr=standard_error[i], ecolor='blue', lw=2.0)
        plt.plot(lag, prob_model[i], c=cl2, label='From Model', lw=2.5, marker='o', markeredgecolor=cl2, markersize=8)
        plt.title('State '+str(i+1), fontsize = fs)
        plt.xlabel('Number of Markov steps', fontsize = fs)
        plt.ylabel('Probability remaining', fontsize = fs)
        plt.tick_params(axis='both', labelsize=fs)
        plt.xlim(0.8, num_steps+0.2)
        plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.show()

def evaluate_ck_pass(predictions, obs_estimates, pred_errors, obs_errors, threshold=1.96):
    """
    Evaluates if the MSM passes the CK test based on Z-scores.
    """
    combined_se = np.sqrt(np.square(pred_errors) + np.square(obs_errors))
    z_scores = np.abs(predictions - obs_estimates) / (combined_se + 1e-10)
    max_z = np.max(z_scores)
    if max_z <= threshold:
        pass_test = True
        note = "prediction within the 95% confidence interval"
    elif max_z <= threshold * 2:
        pass_test = True
        note = "prediction outside the 95% confidence interval but acceptable"
    else:
        pass_test = False
        note = "prediction significantly different from observations"
    return {
        "pass": pass_test,
        "note": note,
        "max_z_score": max_z,
        "z_scores": z_scores.tolist(),
    }
