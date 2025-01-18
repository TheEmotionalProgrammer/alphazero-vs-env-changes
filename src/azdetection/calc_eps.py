def calc_eps(n, gamma, k):
    return 1 - gamma**(n-k)

# if __main__

if __name__ == "__main__":
    print(calc_eps(3, 0.95, 1)) # 0.0297

 