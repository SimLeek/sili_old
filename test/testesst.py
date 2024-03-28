def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def puzzle_recursive(N, k):
    def puzzle_P(k):
        if k == 1:
            return 1 / factorial(N)
        else:
            return (1 - puzzle_recursive(N, k-1)) / factorial(N)

    return sum(puzzle_P(i) for i in range(1, k+1))

print(puzzle_recursive(4,17))