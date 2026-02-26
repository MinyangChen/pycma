import numpy as np

path = r"convergence/ackley20_de_gp_runs5.npz"
d = np.load(path, allow_pickle=True)

print("keys:", d.files)
fe = d["fe"]
mean = d["mean"]
print("fe shape:", fe.shape, "mean shape:", mean.shape)
print("fe head/tail:", fe[0:50], fe[-10:])
print("mean head/tail:", mean[0:50], mean[-5:])

# check whether fe is a uniform grid
diff = np.unique(np.diff(fe))
print("unique step sizes in fe (first 20):", diff[:20], "count:", diff.size)
