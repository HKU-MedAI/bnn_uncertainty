from distributions import gigrnd
import torch

def main():

    p = torch.rand((1, 32, 256))
    p[0, 0, :] = -p[0, 0, :]
    a = 2
    b = 3

    s = gigrnd(p, a, b)

    print(s)

if __name__ == "__main__":
    main()