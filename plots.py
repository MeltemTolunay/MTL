import matplotlib.pyplot as plt
import csv


losses = []
with open('loss_history.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        losses.extend(line)

y = [float(i) for i in losses]
x = [i for i in range(len(y))]

fig1 = plt.figure()
plt.plot(x, y)
plt.title('Cross-Entropy Loss')
plt.xlabel('number of iterations')
plt.ylabel('loss')
plt.savefig('loss.jpg')


accuracies = []
with open('accuracy.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        accuracies.extend(line)

y = [float(i) for i in accuracies]
x = [i for i in range(len(y))]

fig2 = plt.figure()
plt.plot(x, y)
plt.title('Validation Accuracies')
plt.xlabel('number of epochs')
plt.ylabel('validation accuracy')
plt.savefig('acc.jpg')


'''
/Users/Meltem/anaconda3/envs/pytorch04/bin/python /Users/Meltem/PycharmProjects/MTL/altogether.py
Epoch 0/24
----------
train Loss: 0.8884 Acc: 0.4992
val Loss: 0.8314 Acc: 0.5532

Epoch 1/24
----------
train Loss: 0.7914 Acc: 0.5610
val Loss: 0.7118 Acc: 0.5957

Epoch 2/24
----------
train Loss: 0.7119 Acc: 0.6111
val Loss: 0.6860 Acc: 0.6383

Epoch 3/24
----------
train Loss: 0.6831 Acc: 0.6335
val Loss: 0.6337 Acc: 0.7128

Epoch 4/24
----------
train Loss: 0.6381 Acc: 0.6520
val Loss: 0.6146 Acc: 0.7340

Epoch 5/24
----------
train Loss: 0.6035 Acc: 0.6813
val Loss: 0.5929 Acc: 0.7447

Epoch 6/24
----------
train Loss: 0.5917 Acc: 0.6875
val Loss: 0.5596 Acc: 0.7340

Epoch 7/24
----------
train Loss: 0.5487 Acc: 0.7215
val Loss: 0.5696 Acc: 0.7447

Epoch 8/24
----------
train Loss: 0.5564 Acc: 0.6983
val Loss: 0.5507 Acc: 0.7447

Epoch 9/24
----------
train Loss: 0.5711 Acc: 0.7145
val Loss: 0.5684 Acc: 0.7606

Epoch 10/24
----------
train Loss: 0.5546 Acc: 0.7238
val Loss: 0.5607 Acc: 0.7553

Epoch 11/24
----------
train Loss: 0.5577 Acc: 0.7137
val Loss: 0.5783 Acc: 0.7447

Epoch 12/24
----------
train Loss: 0.5497 Acc: 0.7137
val Loss: 0.5407 Acc: 0.7553

Epoch 13/24
----------
train Loss: 0.5481 Acc: 0.7076
val Loss: 0.5500 Acc: 0.7660

Epoch 14/24
----------
train Loss: 0.5585 Acc: 0.7191
val Loss: 0.5412 Acc: 0.7500

Epoch 15/24
----------
train Loss: 0.5576 Acc: 0.7207
val Loss: 0.5530 Acc: 0.7394

Epoch 16/24
----------
train Loss: 0.5716 Acc: 0.7160
val Loss: 0.5421 Acc: 0.7553

Epoch 17/24
----------
train Loss: 0.5544 Acc: 0.7106
val Loss: 0.5589 Acc: 0.7606

Epoch 18/24
----------
train Loss: 0.5535 Acc: 0.7037
val Loss: 0.5548 Acc: 0.7394

Epoch 19/24
----------
train Loss: 0.5464 Acc: 0.7191
val Loss: 0.5625 Acc: 0.7606

Epoch 20/24
----------
train Loss: 0.5507 Acc: 0.7261
val Loss: 0.5673 Acc: 0.7394

Epoch 21/24
----------
train Loss: 0.5394 Acc: 0.7353
val Loss: 0.5476 Acc: 0.7447

Epoch 22/24
----------
train Loss: 0.5401 Acc: 0.7338
val Loss: 0.5551 Acc: 0.7500

Epoch 23/24
----------
train Loss: 0.5343 Acc: 0.7400
val Loss: 0.5516 Acc: 0.7553

Epoch 24/24
----------
train Loss: 0.5571 Acc: 0.7215
val Loss: 0.5431 Acc: 0.7606

Training complete in 62m 26s
Best val Acc: 0.765957
2018-05-10 17:59:22.019 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:22.019 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:23.834 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:23.834 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:24.875 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:24.875 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:26.279 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:26.279 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:28.228 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:28.228 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:28.641 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:28.641 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:31.393 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:31.393 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]
2018-05-10 17:59:37.221 python[3561:247855] *** Assertion failure in -[LUPresenter animationControllerForTerm:atLocation:options:], /SourceCache/Lookup/Lookup-160/Framework/Classes/LUPresenter.m:264
2018-05-10 17:59:37.221 python[3561:247855] Lookup: Unhandled exception 'NSInternalInconsistencyException' caught in -[LULookupDefinitionModule lookupAnimationControllerForString:range:options:originProvider:inView:]

Process finished with exit code 0
'''

