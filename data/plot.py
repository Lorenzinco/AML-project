from matplotlib import pyplot as plt
from json import load

js = load(open("hist.json"))

tra = [x["train_loss"] for x in js]
val = [x["val_loss"] for x in js]
d = [x["d_loss"] for x in js]
plt.plot(tra, label="train")
plt.plot(val, label="val")
plt.plot(d, label="discriminator")
plt.legend()
plt.show()

