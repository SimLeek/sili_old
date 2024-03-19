import math


def _primes_up_to(n):
    """Simple function to get a prime lookup table"""
    primes = []
    for i in range(2, n + 1):
        is_prime = True
        for j in range(2, int(math.ceil(math.sqrt(i)))):
            if i % j == 0:
                is_prime = False
        if is_prime:
            primes.append(i)
    return primes


plut = [2, 3, 4, 5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
        101, 103, 107, 109, 113, 121, 127, 131, 137, 139, 149, 151, 157, 163, 167, 169, 173, 179, 181, 191, 193, 197,
        199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 289, 293, 307, 311, 313, 317,
        331, 337, 347, 349, 353, 359, 361, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449,
        457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 529, 541, 547, 557, 563, 569, 571, 577, 587, 593,
        599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
        739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 841, 853, 857, 859, 863, 877,
        881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 961, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
        1021]


def prime_factors(n):
    """Simple fast prime factorization function for smallish numbers."""
    # todo: make usable for larger numbers using https://stackoverflow.com/a/2274520
    factors = []
    if int(math.ceil(math.sqrt(n))) < plut[-1]:
        for p in plut:
            if p>math.sqrt(n):
                break
            while n % p == 0:
                factors.append(p)
                n = n // p

        if n > 2:  # n is prime
            factors.append(n)
    else:
        raise NotImplementedError("What kind of GPU are you using where this is necessary?")

    return factors


def knapsack_upper(n, D):
    W = int(math.ceil(n ** (1 / D)))
    wt = val = prime_factors(n)

    l = len(val)

    K = [[1 for _ in range(W + 1)] for _ in range(l + 1)]

    # Build table K[][] in bottom up manner
    for i in range(l + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 1
            elif K[i - 1][w] <= w:
                if (wt[i - 1] > W):
                    K[i][w] = K[i - 1][w]
                else:
                    K[i][w] = max(val[i - 1]
                                  * K[i - 1][w - wt[i - 1]],
                                  K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    return K[l][W]


def knapsack_lower(n, D):
    W = int(math.ceil(n ** (1 / D)))
    wt = val = prime_factors(n)

    l = len(val)

    K = [[1 for _ in range(W + 1)] for _ in range(l + 1)]

    # Build table K[][] in bottom up manner
    for i in range(l + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 1
            elif K[i - 1][w] <= w:
                if(wt[i-1]>W):
                    K[i][w]=K[i - 1][w]
                else:
                    K[i][w] = max(val[i - 1]
                                  * K[i - 1][w - wt[i - 1]],
                                  K[i - 1][w])
                if K[i][w] > w:
                    K[i][w] = K[i - 1][w]
            else:
                K[i][w] = K[i - 1][w]

    return K[l][W]


def find_good_dimension_sizes(max_workgroup_size, dimensions):
    add_dims = []
    if dimensions>len(prime_factors(max_workgroup_size)):
        add_dims = [1]*(dimensions-len(prime_factors(max_workgroup_size)))
        dimensions = len(prime_factors(max_workgroup_size))

    remainders = [[max_workgroup_size, []]]
    new_remainders = []
    for d in reversed(range(2, dimensions + 1)):
        for r in range(len(remainders)):
            ku = knapsack_upper(remainders[r][0], d)
            kl = knapsack_lower(remainders[r][0], d)
            new_remainders.append([int(remainders[r][0] / ku), remainders[r][1] + [ku]])
            new_remainders.append([int(remainders[r][0] / kl), remainders[r][1] + [kl]])
        del remainders
        remainders = new_remainders
        new_remainders = []

    for r in range(len(remainders)):
        new_remainders.append(remainders[r][1] + [remainders[r][0]])
    remainders = new_remainders

    min_item = []
    min_val = float('inf')
    for r in remainders:
        avg = sum(r) / len(r)
        val = sum([abs(x - avg) for x in r])
        if val < min_val:
            min_val = val
            min_item = r

    return min_item+add_dims


if __name__ == "__main__":
    print(find_good_dimension_sizes(16536, 3))
