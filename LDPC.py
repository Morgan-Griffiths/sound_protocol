import numpy as np

def compute_initial_LLRs_base10(received_codeword, noise_variance=1.0):
    """
    Compute the initial LLRs for a received codeword over an AWGN channel using 10-PAM.

    :param received_codeword: numpy array, the received codeword
    :param noise_variance: float, the noise variance of the AWGN channel
    :return: numpy array, the initial LLRs (shape: len(received_codeword) x 10)
    """
    symbol_set = np.arange(10)
    LLRs = np.zeros((len(received_codeword), 10))

    for i, r in enumerate(received_codeword):
        for j, s in enumerate(symbol_set):
            probabilities = np.exp(-(r - symbol_set) ** 2 / (2 * noise_variance))
            denominator = np.sum(probabilities)
            probabilities_normalized = probabilities / denominator

            for j in range(10):
                LLRs[i, j] = np.log(probabilities_normalized[j] / probabilities_normalized[0])


    return LLRs



def compute_initial_LLRs(received_codeword, noise_variance=1.0):
    """
    Compute the initial LLRs for a received codeword over an AWGN channel.

    :param received_codeword: numpy array, the received codeword
    :param noise_variance: float, the noise variance of the AWGN channel
    :return: numpy array, the initial LLRs
    """
    LLRs = 2 * received_codeword / noise_variance
    return LLRs



def update_variable_to_check_messages_base10(H, LLRs, check_to_variable_messages):
    """
    Update the messages from variable nodes to check nodes in the base-10 LDPC decoding process.

    :param H: numpy array, the parity-check matrix (shape: m x n)
    :param LLRs: numpy array, the initial LLRs (shape: n x 10)
    :param check_to_variable_messages: numpy array, messages from check nodes to variable nodes (shape: m x n x 10)
    :return: numpy array, updated messages from variable nodes to check nodes (shape: m x n x 10)
    """
    m, n = H.shape
    variable_to_check_messages = np.zeros((m, n, 10))

    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                variable_to_check_messages[i, j] = LLRs[j] - check_to_variable_messages[i, j]

    return variable_to_check_messages

def update_check_to_variable_messages_base10(H, variable_to_check_messages):
    """
    Update the messages from check nodes to variable nodes in the base-10 LDPC decoding process.

    :param H: numpy array, the parity-check matrix (shape: m x n)
    :param variable_to_check_messages: numpy array, messages from variable nodes to check nodes (shape: m x n x 10)
    :return: numpy array, updated messages from check nodes to variable nodes (shape: m x n x 10)
    """
    m, n = H.shape
    check_to_variable_messages = np.ones((m, n, 10))

    for i in range(m):
        for j in range(n):
            if H[i, j] == 1:
                # Compute the sum of the messages from all connected variable nodes except j
                message_sum = np.zeros(10)
                for k in range(n):
                    if H[i, k] == 1 and k != j:
                        message_sum += variable_to_check_messages[i, k]

                # Update the message from check node i to variable node j
                check_to_variable_messages[i, j] = message_sum

    return check_to_variable_messages

def update_variable_beliefs_base10(LLRs, check_to_variable_messages):
    """
    Update the beliefs of variable nodes in the base-10 LDPC decoding process.

    :param LLRs: numpy array, the initial LLRs (shape: n x 10)
    :param check_to_variable_messages: numpy array, messages from check nodes to variable nodes (shape: m x n x 10)
    :return: numpy array, updated beliefs of variable nodes (shape: n x 10)
    """
    n = LLRs.shape[0]
    variable_beliefs = np.zeros((n, 10))

    for j in range(n):
        variable_beliefs[j] = LLRs[j] + np.sum(check_to_variable_messages[:, j], axis=0)

    return variable_beliefs


def check_parity_conditions_base10(H, variable_beliefs):
    """
    Check whether the current estimates of variable nodes satisfy the parity-check conditions in the base-10 LDPC decoding process.

    :param H: numpy array, the parity-check matrix (shape: m x n)
    :param variable_beliefs: numpy array, beliefs of variable nodes (shape: n x 10)
    :return: bool, True if parity-check conditions are satisfied, False otherwise
    """
    m, n = H.shape
    variable_estimates = np.argmax(variable_beliefs, axis=1)

    for i in range(m):
        check_sum = 0
        for j in range(n):
            if H[i, j] == 1:
                check_sum += variable_estimates[j]
        if check_sum % 10 != 0:
            return False

    return True

def make_hard_decisions_base10(variable_beliefs):
    """
    Compute the hard decisions for variable nodes in the base-10 LDPC decoding process.

    :param variable_beliefs: numpy array, beliefs of variable nodes (shape: n x 10)
    :return: numpy array, hard decisions for variable nodes (shape: n)
    """
    return np.argmax(variable_beliefs, axis=1)

def decode_ldpc(received_codeword, H, max_iterations):
    """
    Decode the received codeword using the BP algorithm and the binary parity-check matrix H.

    :param received_codeword: numpy array, the received codeword with possible errors
    :param H: numpy array, the binary parity-check matrix
    :param max_iterations: int, the maximum number of iterations for the BP algorithm
    :return: numpy array, the decoded message
    """
    # Compute the initial LLRs for the received codeword
    LLRs = compute_initial_LLRs_base10(received_codeword)

    check_to_variable_messages = np.random.randn(H.shape[0], H.shape[1], 10)
    # Perform iterative message-passing
    for _ in range(max_iterations):
        # Update messages from variable nodes to check nodes
        check_to_variable_messages = update_variable_to_check_messages_base10(H,LLRs,check_to_variable_messages)

        # Update messages from check nodes to variable nodes
        check_to_variable_messages = update_check_to_variable_messages_base10(H, check_to_variable_messages)

        # Update the variable nodes' beliefs
        LLRs = update_variable_beliefs_base10(LLRs, H)

        # Check if the estimated codeword satisfies the parity-check equations
        if check_parity_conditions_base10(H, LLRs):
            break

    # Make hard decisions on the variable nodes' beliefs
    decoded_message = make_hard_decisions_base10(LLRs)

    return decoded_message


def encode_ldpc_base10(message, G):
    """
    Encode the message using the base-10 generator matrix G.

    :param message: numpy array, a base-10 message vector of size k
    :param G: numpy array, the base-10 generator matrix of size k x n
    :return: numpy array, the encoded message (codeword) of size n
    """
    encoded_message = np.mod(np.dot(message, G), 10)
    return encoded_message


def compute_generator_matrix(H):
    m, n = H.shape
    H_permuted, P = permute_to_standard_form(H)
    A = H_permuted[:, :n - m]
    I = np.identity(n - m, dtype=int)
    G_permuted = np.hstack((I, A.T))
    P_inv = np.linalg.inv(P)
    G = (G_permuted @ P_inv) % 10
    return G

def permute_to_standard_form(H):
    m, n = H.shape
    P = np.identity(n, dtype=int)
    for i in range(m):
        if H[i, i] == 0:
            for j in range(i + 1, n):
                if H[i, j] != 0:
                    H[:, [i, j]] = H[:, [j, i]]
                    P[:, [i, j]] = P[:, [j, i]]
                    break
    return H, P

def find_full_rank_G():
    while True:
        H = np.random.randint(0, 10, size=(3, 6))
        G = compute_generator_matrix(H)
        print(np.linalg.matrix_rank(G) == 10)
        if len(np.unique(G)) == 10:
            break
    return H

# Example usage
H = np.array([[2,7,8,4,9,5],
            [5,3,0,4,0,7],
            [6,9,4,2,1,5]])

G = compute_generator_matrix(H)
print(G,G.shape)
# LLRs = np.random.randn(6, 10)
# check_to_variable_messages = np.random.randn(3, 6, 10)

# # Define your message as a base-10 vector
message_base10 = np.array([0, 7, 2])

# # Encode the message using the base-10 generator matrix G
print('message_base10',message_base10)
codeword_base10 = encode_ldpc_base10(message_base10, G)
print("Encoded base-10 message (codeword):", codeword_base10,type(codeword_base10[0]))


decoded_message = decode_ldpc(codeword_base10,H,20)
print("Decoded message:", decoded_message)