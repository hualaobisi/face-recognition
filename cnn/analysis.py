import matplotlib.pyplot as plt

epoch = list(range(200))
test_acc1 = []
test_acc2 = []
test_acc3 = []
test_acc4 = []
time1 = []
time2 = []
time3 = []
time4 = []
loss1 = []
loss2 = []
loss3 = []
loss4 = []

for line in open("./result/acc_lenet_noBN.txt","r"):
     line = line[:-1]
     list1 = line.replace('=',',').strip().split(",")
     test_acc1.append(float(list1[3].strip('%'))/100)
     time1.append(float(list1[-1].strip('%')))


for line2 in open("./result/acc_lenet_BN.txt","r"):
     line2 = line2[:-1]
     list2 = line2.replace('=',',').strip().split(",")
     test_acc2.append(float(list2[3].strip('%'))/100)
     time2.append(float(list2[-1].strip('%')))

for line3 in open("./result/acc_resnet18_nodrop.txt","r"):
     line3 = line3[:-1]
     list3 = line3.replace('=',',').strip().split(",")
     test_acc3.append(float(list3[3].strip('%'))/100)
     time3.append(float(list3[-1].strip('%')))

for line4 in open("./result/acc_resnet18_dropblock.txt","r"):
     line4 = line4[:-1]
     list4 = line4.replace('=',',').strip().split(",")
     test_acc4.append(float(list4[3].strip('%'))/100)
     time4.append(float(list4[-1].strip('%')))




# Lenet test acc
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(epoch,test_acc1,label="LeNet5_noBN_learning_rate = 0.001")
ax1.plot(epoch,test_acc2,label="LeNet5_BN_learning_rate = 0.001")
# ax1.plot(epoch,test_acc3,label="ResNet18_nodrop_learning_rate = 0.01")
ax1.set_ylabel('Test accuracy')
ax1.set_xlabel("Epoch")

fig1.legend()

fig1.set_size_inches(10,8)

plt.savefig('./LeNet5 acc.png')
plt.show()

# Resnet test acc
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(epoch,test_acc3,label="ResNet18_nodrop")
ax2.plot(epoch,test_acc4,label="ResNet18_dropblock")
ax2.set_ylabel('Test accuracy')
ax2.set_xlabel("Epoch")

fig2.legend()

fig2.set_size_inches(10,8)

plt.savefig('./ResNet18 acc.png')
plt.show()

# Time cost
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(epoch,time1,label="LeNet5_noBN")
ax3.plot(epoch,time2,label="LeNet5_BN")
ax3.plot(epoch,time3,label="ResNet18_nodrop")
ax3.plot(epoch,time4,label="ResNet18_dropblock")
ax3.set_ylabel('Time')
ax3.set_xlabel("Epoch")


print('LeNet5 no BN avg time:',sum(time1)/len(time1))
print('LeNet5 no BN sum time:',sum(time1))
print('LeNet5 BN avg time:',sum(time2)/len(time2))
print('LeNet5 BN sum time:',sum(time2))
print('Resnet18 no drop avg time:',sum(time3)/len(time3))
print('Resnet18 no drop sum time:',sum(time3))
print('Resnet18 dropblock avg time:',sum(time4)/len(time4))
print('Resnet18 dropblock sum time:',sum(time4))

fig3.legend()

fig3.set_size_inches(10,8)

plt.savefig('./Time cost.png')
plt.show()

# train loss and acc

train_acc1 = []
train_acc2 = []
train_acc3 = []
train_acc4 = []
loss1 = []
loss2 = []
loss3 = []
loss4 = []

for line in open("./result/log_lenet_noBN.txt","r"):
     line = line[3:]
     list1 = line.replace(':','|').strip().split("|")
     train_acc1.append(float(list1[-1].strip('%'))/100)
     loss1.append(float(list1[2]))


for line in open("./result/log_lenet_BN.txt","r"):
     line = line[3:]
     list2 = line.replace(':','|').strip().split("|")
     train_acc2.append(float(list2[-1].strip('%'))/100)
     loss2.append(float(list2[2]))

for line in open("./result/log_resnet18_nodrop.txt","r"):
     line = line[3:]
     list3 = line.replace(':','|').strip().split("|")
     train_acc3.append(float(list3[-1].strip('%'))/100)
     loss3.append(float(list3[2]))

for line in open("./result/log_resnet18_dropblock.txt","r"):
     line = line[3:]
     list4 = line.replace(':','|').strip().split("|")
     train_acc4.append(float(list4[-1].strip('%'))/100)
     loss4.append(float(list4[2]))

length = len(loss1)
iter = list(range(length))

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(iter,train_acc1,label="LeNet5_noBN")
ax4.plot(iter,train_acc2,label="LeNet5_BN")
ax4.plot(iter,train_acc3,label="ResNet18_nodrop")
ax4.plot(iter,train_acc4,label="ResNet18_dropblock")
ax4.set_ylabel('Train accuracy')
ax4.set_xlabel("iter")

fig4.legend()
fig4.set_size_inches(10,8)

plt.savefig('./Train acc.png')
plt.show()

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.plot(iter,loss1,label="LeNet5_noBN")
ax5.plot(iter,loss2,label="LeNet5_BN")
ax5.plot(iter,loss3,label="ResNet18_nodrop")
ax5.plot(iter,loss4,label="ResNet18_dropblock")
ax5.set_ylabel('Loss')

fig5.legend()

fig5.set_size_inches(10,8)

plt.savefig('./Train loss.png')
plt.show()