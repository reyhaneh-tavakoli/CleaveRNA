import matplotlib.pyplot as plt

# Read prob file
positions = []
pp = []
lunp = []

with open("HPV.fasta.prob") as f:
    next(f)  # skip header
    for line in f:
        cols = line.strip().split()
        positions.append(int(cols[0]))
        pp.append(float(cols[1]))
        lunp.append(float(cols[2]))

plt.figure(figsize=(15,4))
plt.plot(positions, pp, label='pp', color='blue')
plt.plot(positions, lunp, label='lunp', color='red')
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title("RNA Base-Pairing and Unpaired Probability")
plt.legend()
plt.tight_layout()
plt.savefig("HPV_pp_lunp.png")
plt.show()

