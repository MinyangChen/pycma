import numpy as np

path = r"C:\Users\2948856C\OneDrive - University of Glasgow\0_PHD\2Project\6_highd_opt\2_pycma\pycma-1\convergence\ackley_dts_cma_runs1.npz"
d = np.load(path, allow_pickle=True)

print("keys:", d.files)
fe = d["fe"]
mean = d["mean"]
print("fe shape:", fe.shape, "mean shape:", mean.shape)
print("fe head/tail:", fe[:10], fe[-10:])
print("mean head/tail:", mean[:5], mean[-5:])

# check whether fe is a uniform grid
diff = np.unique(np.diff(fe))
print("unique step sizes in fe (first 20):", diff[:20], "count:", diff.size)
