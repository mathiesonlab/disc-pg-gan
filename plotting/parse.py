'''
Utils for use in seaborn_plot.py. These are mostly taken from ss_helpers.py, 
but used separately to avoid the necessity of activating conda.

Authors: Sara Mathieson, Rebecca Riley
'''

def parse_mini_lst(mini_lst):
    return [float(x.replace("[",'').replace("]",'').replace(",",'')) for x in
        mini_lst]

def add_to_lst(total_lst, mini_lst):
    assert len(total_lst) == len(mini_lst)
    for i in range(len(total_lst)):
        total_lst[i].append(mini_lst[i])

def parse_output(filename, return_acc=False):
    """Parse pg-gan output to find the inferred parameters"""

    def clean_param_tkn(s, cast_int=False):
        if s == 'None,':
            return None # this is a common result (not an edge case)
        
        if s[:-1].isnumeric(): # probably the seed
             # no need to remove quotation marks, just comma
            return s[:-1] # only used as a label, so ok to leave as str
            
        result = s[1:-2]
        if cast_int:
            return int(result)
        return result

    f = open(filename,'r')

    # list of lists, one for each param
    param_lst_all = []

    # evaluation metrics
    disc_loss_lst = []
    real_acc_lst = []
    fake_acc_lst = []

    num_param = None

    trial_data = {}

    for line in f:

        if line.startswith("{"):
            tokens = line.split()
            print(tokens)
            param_str = tokens[3][1:-2]
            print("PARAMS", param_str)
            param_names = param_str.split(",")
            num_param = len(param_names)
            for i in range(num_param):
                param_lst_all.append([])

            trial_data['model'] = clean_param_tkn(tokens[1])
            trial_data['params'] = param_str
            trial_data['data_h5'] = clean_param_tkn(tokens[5])
            trial_data['bed_file'] = clean_param_tkn(tokens[7])
            trial_data['reco_folder'] = clean_param_tkn(tokens[9])
            trial_data['pop'] = trial_data["data_h5"].split("/")[5][0:3] \
                if trial_data["data_h5"] is not None else None
            trial_data['seed'] = clean_param_tkn(tokens[15])
            trial_data['sample_size'] = clean_param_tkn(tokens[17], cast_int=True)
            trial_data['disc'] = clean_param_tkn(tokens[-1])
        
        elif "Epoch 100" in line:
            tokens = line.split()
            disc_loss = float(tokens[3][:-1])
            real_acc = float(tokens[6][:-1])/100
            fake_acc = float(tokens[9])/100
            disc_loss_lst.append(disc_loss)
            real_acc_lst.append(real_acc)
            fake_acc_lst.append(fake_acc)

        if "T, p_accept" in line:
            tokens = line.split()
            # parse current params and add to each list
            mini_lst = parse_mini_lst(tokens[-1-num_param:-1])
            add_to_lst(param_lst_all, mini_lst)

    f.close()

    # Use -1 instead of iter for the last iteration
    final_params = [param_lst_all[i][-1] for i in range(num_param)]
    if return_acc:
        return final_params, disc_loss_lst, real_acc_lst, fake_acc_lst, \
            trial_data
    else:
        return final_params, trial_data
