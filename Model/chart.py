import json

import matplotlib.pyplot as plt
with open("123.json") as file_obj:
     history = json.loads(json.load(file_obj))
epoch = []
acc = []
loss = []
for i in history:
    epoch.append(i['epoch'])
    acc.append(i['acc'])
    loss.append(i['loss'])
print(epoch)
print(acc)
print(loss)
fig1, ax_acc = plt.subplots()
plt.grid(True)
plt.plot(acc, 'r', label='train acc')
# plt.plot(history.history['val_accuracy'], 'k', label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model-Accuracy')
plt.legend(['training', 'validation'], loc='lower right')
plt.savefig('accuracy.png', format='png')  # 保存结果
plt.show()

fig2, ax_loss = plt.subplots()
plt.grid(True)
plt.plot(history.history['loss'], 'g', label='train loss')
plt.plot(history.history['val_loss'], 'k', label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model-Loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.savefig('loss.png', format='png')  # 保存结果
plt.show()
