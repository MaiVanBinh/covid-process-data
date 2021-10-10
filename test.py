import pickle

history = pickle.load(open('./VGG-16/trainHistoryDict', "rb"))
print(history.keys())
print(history['acc'])
[
    ['012', 'A B', 'A', 'B'],
    ['014', 'A D', 'A', 'C'],
]