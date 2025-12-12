from matplotlib import pyplot as plt
from json import load

js = load(open("loss_history.json"))

tra = [x["train_loss"] for x in js]
val = [x["val_loss"] for x in js]
plt.plot(val, label="val")
plt.plot(tra, label="train")
plt.legend()
plt.show()

