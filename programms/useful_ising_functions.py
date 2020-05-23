import numpy as np
import time
import concurrent.futures as ftr
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sys import exit

def key_array(length, rank):
    qq = np.zeros((length,length))
    ss = np.ones(length)
    qq[rank] = ss
    return qq

def sigmoid(x):
    x = np.clip(x, -9.0, 9.0)
    return 1/(1 + np.exp(-x))

def soft_max(x):
    if x.ndim == 1:
        z = np.sum(np.exp(x))
        return   np.exp(x)/z
    x = np.clip(x, -9.0, 9.0)
    z = np.sum(np.exp(x), axis = 1).reshape(-1, 1)
    return   np.exp(x)/z
# -----------------------------------------------------------------------------------
# -----------------------KL_Divergence_batch_friendly--------------------------------
# -----------------------------------------------------------------------------------
def KL(target, prediction):

    if prediction.ndim == 1:
        prediction = prediction.reshape(1, prediction.shape[0])
    if target.ndim == 1:
        target = target.reshape(1, target.shape[0])
    if np.all(target.shape != prediction.shape):
        print("KL divergence needs inputs that are the same array shape. Terminating program...")
        exit()

    N_batch = prediction.shape[0]
    #adding a small value so as to process values that were 0
    small = 0.01
    target += small
    prediction += small
    #Actual calculation
    KL_output = np.sum(target*(np.log2(np.clip(target/prediction, 0.00000001, 1000000))), axis = 1)
    KL_output = np.sum(KL_output)/N_batch
    return KL_output


# -----------------------------------------------------------------------------------
# ----------------------one_hot_representation_for_nn--------------------------------
# -----------------------------------------------------------------------------------
def one_hot_representation(T_list, H_list, T0, H0):
    output = np.zeros((T_list.shape[0])*(H_list.shape[0]))
    N = 0
    for T in T_list:
        for H in H_list:
            if T == T0 and H == H0:
                output[N] = 1
            N += 1
    if (output == np.zeros((T_list.shape[0])*(H_list.shape[0]))).all():
        print("No such state in the lists...")
        quit()
    return output

def one_hot_representation_from_id(T_list, H_list, Events, id):
    batch_size = id.shape[0]
    output = np.zeros((batch_size, (T_list.shape[0])*(H_list.shape[0])))
    step = int(Events/2)
    N = divmod(id, step)[0]
    for bb in range(batch_size):
        output[bb][N[bb]] = 1
    return output

# -----------------------------------------------------------------------------------
# ----------------------Correlation_vector_for_calculations--------------------------
# -----------------------------------------------------------------------------------
def corr(A,B):
    Length = A.shape[0]
    R = np.zeros(Length)
    for ll in range(Length):
        R += np.roll(np.matmul(A,(B)*key_array(Length,ll))[ll], -ll)
        R += np.roll(np.matmul(A,(B).transpose()*key_array(Length,ll))[ll], -ll)
    R /= 2*Length
    return R

# -----------------------------------------------------------------------------------
# -------------------Makes_a_RG_transformation_to_L/2_state--------------------------
# -----------------------------------------------------------------------------------
def RG_crunch(state):
    Length = state.shape[0]
    crunched_state = np.zeros((int(Length/2), int(Length/2)))
    for i in 2*np.arange(5):
        for j in 2*np.arange(5):
            crunched_state[int(i/2),int(j/2)] = np.sign(state[i,j] + state[i,j+1] + state[i+1,j] + state[i+1,j+1])
            if crunched_state[int(i/2),int(j/2)] == 0:
                p = np.random.rand(1)
                if p > 0.5:
                    crunched_state[int(i/2),int(j/2)] = 1
                else:
                    crunched_state[int(i/2),int(j/2)] = -1

    return crunched_state

# -----------------------------------------------------------------------------------
# -------------------Important_for_RBM_storage_functions-----------------------------
# -----------------------------------------------------------------------------------
def change_to_binary(input):
    return (input + 1)/2
def return_from_binary(input):
    return (2*input) -1

# -----------------------------------------------------------------------------------
# ---------------------Does_one_Accelerated_Metropolis_Swipe-------------------------
# -----------------------------------------------------------------------------------
def metro_swipe(input_matrix, temperature, ext_field):
    s = input_matrix.copy()
    k = 1 / temperature
    ly = s.shape[0]
    lx = s.shape[1]

    # flip light squares
    flip_prob = np.exp(2 * interactions(s, k, ext_field))
    rnd = np.random.rand(ly, lx)
    flipper = np.sign(np.floor(rnd - flip_prob)).astype(int)
    flipper[1::2, ::2] = 0  # Clean even fields in odd rows
    flipper[::2, 1::2] = 0  # Clean odd fields in even rows
    first_step = 2 * flipper + 1
    s *= first_step

    # flip dark squares
    flip_prob = np.exp(2 * interactions(s, k, ext_field))
    flipper = np.sign(np.floor(rnd - flip_prob)).astype(int)
    flipper[1::2, 1::2] = 0  # Clean odd fields in odd rows
    flipper[::2, ::2] = 0  # Clean even fields in even rows
    second_step = 2 * flipper + 1
    s *= second_step

    return s

# -----------------------------------------------------------------------------------
# ------------------Creates_a_Matrix_with_the_Energy_of_Each_Site--------------------
# -----------------------------------------------------------------------------------
def interactions(input_matrix, coupling, ext_field):
    lx = input_matrix.shape[0]
    ly = input_matrix.shape[1]

    dbl_diag = np.eye(lx, ly, k=1, dtype=int) + np.eye(lx, ly, k=-1, dtype=int)
    dbl_diag[0][-1] = 1
    dbl_diag[-1][0] = 1

    nnb = np.dot(input_matrix, dbl_diag) + np.dot(dbl_diag, input_matrix)
    ints = - coupling *( input_matrix * nnb + ext_field * input_matrix)

    return ints

# ------------------------------------------------------------------------------------
# -------------------Calculates_Configuration's_Magnetization-------------------------
# ------------------------------------------------------------------------------------
def magnetization(input_matrix):
    s = input_matrix.copy()
    ly = s.shape[0]
    lx = s.shape[1]
    mag = np.sum(s) / (lx * ly)
    return mag

# -----------------------------------------------------------------------------------
# -----------------------Makes_a_Yes_or_No_question_to_User--------------------------
# -----------------------------------------------------------------------------------
def yes_or_no(question):
    while True:
        answer = input(question + '(y/n)')
        if   answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            pass
#-----------------------------------------------------------------------------------
#-------Goes_inside_existing_big_folder_and_makes_or_goes_inside_small_folder-------
#-----------------------------------------------------------------------------------
def RBM_folder(big_folder, small_folder, create = True):
    dir = os.getcwd()
    os.chdir(dir+"/"+big_folder)
    try:
        if create:
            os.makedirs(dir+"/"+big_folder+"/"+small_folder)
        else: print("Didn't create")
    except OSError:
        pass
    finally:
        os.chdir(dir+"/"+big_folder+"/"+small_folder)

#-----------------------------------------------------------------------------------
#--------------------Makes_a_text_file_of_T_H_L_Events_np.array---------------------
#-----------------------------------------------------------------------------------
def RBM_make_text(Length, Events, Temp, Field, thermalization, decorrelation):

        if not(os.path.exists(f'T_{Temp}_H_{Field}.txt')):
            key = True
        else:
            key = yes_or_no(f"T_{Temp}_H_{Field}.txt already exists, do you want to overwrite it?")

        if key:

            #cold_start
            S = np.ones((Length, Length), dtype=int)
            #thermalization
            for tht in range(thermalization):
                S = metro_swipe(S, Temp, Field)
            #making_of_huge_matrix_(Events, Length*Length)
            huge = np.zeros(Length*Length)
            for _ in range(Events):
                huge = np.vstack((huge, S.flatten()))
                #decorrelation
                for dct in range(decorrelation):
                    S = metro_swipe(S, Temp, Field)
            #getting_rid_of_the_initialized_zeros
            output = np.delete(huge, 0, 0)
            huge = None
            del huge
            #saving_state_in_txt
            #np.savetxt(f'RBM_Data_T_{Temp}_H_{Field}.txt', output)
            save_RBM_State_data(f'T_{Temp}_H_{Field}.txt', output, question = False)
        else:
            pass


# -----------------------------------------------------------------------------------
# --------Converts_a_line_from_text_file_to_flattened_state_representation-----------
# -----------------------------------------------------------------------------------
def RBM_line_reader(line_to_be_read_from_text_file, L):
    list = []
    N = 0
    for l in line_to_be_read_from_text_file:
        N += 1
        if N < (L*L) + 1:
            list.append(l)
    list = [ int(x) for x in list ]
    output = np.asarray(list)
    return return_from_binary(output)
# -----------------------------------------------------------------------------------
# -----------------Saves_Data_in_form_of_np.array_in_specified_file_in_cwd-----------
# -----------------------------------------------------------------------------------
def save_RBM_State_data(file, array, question = True):
    answer = True
    if os.path.exists(file) and question:
        answer = yes_or_no("Do you want to overwrite "+file)
    if answer == True:
        np.savetxt(file, array)
    else: print("The text file was left intact")
# -----------------------------------------------------------------------------------
# -----------Loads_Data_in_form_of_np.array_in_specified_existing_file_in_cwd--------
# -----------------------------------------------------------------------------------
def load_RBM_State_data(file):
    answer = True
    if not os.path.exists(file):
        answer = False
    if answer:
        array = np.loadtxt(file)
        return array
    else:
        print("No such file in directory..."+file+"Terminating the programm.")
        exit()
# -----------------------------------------------------------------------------------
# -----------------Divides_data_for_testing_training_nn------------------------------
# -----------------------------------------------------------------------------------
def test_train_divide(array):
    N = 0
    col1 = []
    col2 = []
    for state in array:
        if divmod(N, 2)[1] == 0:
            col1.append(state)
        else:
            col2.append(state)
        N += 1
    train = np.vstack(col1)
    test = np.vstack(col2)
    return (test, train)
# -----------------------------------------------------------------------------------
# -----------Makes_batches_corresponding_id_lists_shuffled_or_not--------------------
# -----------------------------------------------------------------------------------
def batch_maker(data_unbatched, batch_size, seed = 50, random = True, random_seed = False):

    #data shuffled and their original place is shuffled in the same way.
    N_states = data_unbatched.shape[0]
    batch_number = int(N_states/batch_size)
    id = np.linspace(0, N_states - 1, N_states, dtype = int)

    if random_seed:
        rnd_seed = np.random.randint(1, 50)
        np.random.seed(rnd_seed)
        np.random.shuffle(data_unbatched)
        np.random.seed(rnd_seed)
        np.random.shuffle(id)

    else:
        if random:
            np.random.seed(seed)
            np.random.shuffle(data_unbatched)
            np.random.seed(seed)
            np.random.shuffle(id)
        else:
            print("not Shuffled")
    batches = np.split(data_unbatched, batch_number, axis = 0)
    ids = np.split(id, batch_number, axis = 0)

    return batches, ids


class my_simple_nn:

    version = "3.5"

    def __init__(self, N_visible, N_hidden, N_output, learning_rate, learning_momentum):

        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.N_visible = N_visible
        self.N_hidden = N_hidden
        self.N_output = N_output
        self.W_hidden = 10*np.random.rand(N_visible, N_hidden) - 5
        self.W_output = 10*np.random.rand(N_hidden, N_output) - 5
        self.bias_hidden = 10*np.random.rand(N_hidden) - 5
        self.bias_output = 10*np.random.rand(N_output) - 5


    def predict(self, visible_layer, max_repr = False):

        output = np.matmul(visible_layer, self.W_hidden) + self.bias_hidden
        output = np.tanh(output)
        output = np.matmul(output, self.W_output) + self.bias_output
        output = np.tanh(output)
        output = soft_max(output)

        if max_repr:
            N_id = np.argmax(output, axis = 1)
            batch_size = N_id.shape[0]
            output = np.zeros((batch_size, output.shape[1]))
            for bb in range(batch_size):
                output[bb][N_id[bb]] = 1
        return output


    def train(self, visible_layer ,desired_state, dWo, dbo, dWh, dbh):

        N_batch = visible_layer.shape[0]

        hidden = np.matmul(visible_layer, self.W_hidden) + self.bias_hidden
        hidden = np.tanh(hidden)
        pi = np.matmul(hidden, self.W_output) + self.bias_output
        pi = np.tanh(pi)
        output = soft_max(pi)

        u = (1 - (pi**2))*(desired_state - output)
        y = (1 - (hidden**2))*(np.matmul(u, self.W_output.transpose()))

        dWo = ((self.learning_rate/N_batch)*np.tensordot(hidden, u, axes = (0, 0))) + (self.learning_momentum*dWo)
        dbo = ((self.learning_rate/N_batch)*np.sum(u, axis = 0)) + (self.learning_momentum*dbo)
        dWh = ((self.learning_rate/N_batch)*np.tensordot(visible_layer, y, axes = (0, 0))) + (self.learning_momentum*dWh)
        dbh = ((self.learning_rate/N_batch)*np.sum(y, axis = 0)) + (self.learning_momentum*dbh)

        self.W_output += dWo
        self.bias_output += dbo
        self.W_hidden += dWh
        self.bias_hidden += dbh

        return [dWo, dbo, dWh, dbh]

class group_thermometer:

    version = "1.0"

    def __init__(self, N_visible, N_hidden, N_output, learning_rate, learning_momentum, mode = "tanh"):

        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.N_visible = N_visible
        self.N_hidden = N_hidden
        self.N_output = N_output
        self.W_hidden = 2*np.random.rand(N_visible, N_hidden) - 1
        self.W_output = 2*np.random.rand(N_hidden, N_output) - 1
        self.bias_hidden = 2*np.random.rand(N_hidden) - 1
        self.bias_output = 2*np.random.rand(N_output) - 1
        self.mode = mode

    def predict(self, visible_layer):

        if visible_layer.ndim == 1:

            visible_layer = visible_layer.reshape(1, visible_layer.shape[0])

        N_group = visible_layer.shape[0]

        output = np.matmul(visible_layer, self.W_hidden) + self.bias_hidden
        if self.mode == "tanh":
            output = np.tanh(output)
        elif self.mode == "sigmoid":
            output = sigmoid(output)
        else:
            print("Needs an already existing mode. Terminating programm")
            exit()
        output = np.sum(output, axis = 0)/N_group
        output = np.matmul(output, self.W_output) + self.bias_output
        output = np.tanh(output)
        output = soft_max(output)
        return output

    def train(self, visible_layer, desired_state, dWo, dbo, dWh, dbh):

        N_group = visible_layer.shape[0]

        hidden = np.matmul(visible_layer, self.W_hidden) + self.bias_hidden
        if self.mode == "tanh":
            hidden = np.tanh(hidden)
        elif self.mode == "sigmoid":
            hidden = sigmoid(hidden)
        else:
            print("Needs an already existing mode. Terminating programm")
            exit()


        phi = np.sum(hidden, axis = 0)/N_group
        alpha = np.matmul(phi, self.W_output) + self.bias_output
        alpha = np.tanh(alpha)
        output = soft_max(alpha)

        u = (1 - (alpha**2))*(desired_state - output)
        if self.mode == "tanh":
            y = (np.matmul(visible_layer.transpose(), (1 - (hidden)**2)))/N_group
        elif self.mode == "sigmoid":
            y = (np.matmul(visible_layer.transpose(), (hidden - (hidden)**2)))/N_group
        else:
            print("Needs an already existing mode. Terminating programm")
            exit()
        d = np.matmul(self.W_output, u.transpose())


        dWo = (self.learning_rate*np.outer(phi, u)) + (self.learning_momentum*dWo)
        dbo = (self.learning_rate*u) + (self.learning_momentum*dbo)
        dWh = (self.learning_rate*(y*d)) + (self.learning_momentum*dWh)
        if self.mode == "tanh":
            dbh = (self.learning_rate/N_group)*np.sum((1 - (hidden**2))*d, axis = 0) + (self.learning_momentum*dbh)
        elif self.mode == "sigmoid":
            dbh = (self.learning_rate/N_group)*np.sum((hidden - (hidden**2))*d, axis = 0) + (self.learning_momentum*dbh)
        else:
            print("Needs an already existing mode. Terminating programm")
            exit()

        self.W_output += dWo
        self.bias_output += dbo
        self.W_hidden += dWh
        self.bias_hidden += dbh

        return [dWo, dbo, dWh, dbh]

class my_RBM:

    version = "2.0"

    def __init__(self, N_visible, N_hidden, learning_rate):

        self.N_visible = N_visible
        self.N_hidden = N_hidden
        self.learning_rate = learning_rate

        #theta parameters initialized

        self.W = 2*np.random.rand(N_visible, N_hidden) - 1
        self.bias_visible = 2*np.random.rand(N_visible) - 1
        self.bias_hidden = 2*np.random.rand(N_hidden) - 1

    def forward(self, visible_layer):

        N_batch = visible_layer.shape[0]
        rand_h = np.random.rand(N_batch, self.N_hidden)
        prob_h = sigmoid(np.matmul(visible_layer,self.W) + self.bias_hidden)
        return np.sign(prob_h - rand_h)

    def backward(self, hidden_layer):

        N_batch = hidden_layer.shape[0]
        rand_v = np.random.rand(N_batch, self.N_visible)
        prob_v = sigmoid(np.matmul(hidden_layer,(self.W).transpose()) + self.bias_visible)
        return np.sign(prob_v - rand_v)


    def reconstruct(self, visible_layer):
        return self.backward(self.forward(visible_layer))

    def train(self, visible_layer):

        old_vis = visible_layer
        N_batch = old_vis.shape[0]
        old_hid = self.forward(old_vis)
        new_vis = self.backward(old_hid)
        new_hid = self.forward(new_vis)

        #gradient calculation

        dbv = np.average(old_vis - new_vis, axis=0)
        dbh = np.average(old_hid - new_hid, axis=0)
        dW = (np.tensordot(old_vis, old_hid, axes=(0, 0)) / N_batch) - (np.tensordot(new_vis, new_hid, axes=(0, 0)) / N_batch)

        # updating of theta parameters
        self.W += dW * self.learning_rate
        self.bias_visible += dbv * self.learning_rate
        self.bias_hidden += dbh * self.learning_rate

    def loss(self, target):
        #this fuction is to calculate mean error and rsme of wrong reconstructed spins for a batch.
        reconstructed = self.reconstruct(target)
        N_batch = target.shape[0]

        mean = np.sum(np.abs(reconstructed - target))/(2*N_batch)

        cov = np.sum(np.abs(reconstructed - target), axis = 1)/2
        cov = np.sum(cov**2)/N_batch
        cov -= mean**2
        cov = np.sqrt(cov)

        return mean, cov

    def flow(self, flow, iterations):

        for _ in range(iterations):

            flow = self.reconstruct(flow)

        return flow
