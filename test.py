import pickle

history = pickle.load(open('./VGG-16/trainHistoryDict', "rb"))
print(history)
