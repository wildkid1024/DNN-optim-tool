# DNN-optim-tool

This is a fast and flexible decoupled DNN framework which makes computer optimization researchers implement optimization simply and friendly!

## Requirement

Make sure you have install the packages as follow before you start:

- pytorch 
- numpy
- pandas

## Usage

Make sure that you know a little about python and nothing about pytorch,okay,this is a joke.

Firstly,it assume that the network is a multilayer perceptron(MLP) and the dataset is MNIST,maybe you should learn a little about neural network.Let's define a flexible mlp network now, it allows you define a any size and any number of layers mlp network. Just do it as you like:

```python
model1 = MLP(layer_size=1024, layer_num=10) # create a 1024 * 10 mlp network
print(model1)

model2 = MLP() # create a default mlp network(256 * 3)  
print(model2)
```
You will get the output:
```bash
# For model1
MLP(
  (hidden_1): Linear(in_features=784, out_features=1024, bias=True)
  (hidden_2): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_3): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_4): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_5): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_6): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_7): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_8): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_9): Linear(in_features=1024, out_features=1024, bias=True)
  (hidden_10): Linear(in_features=1024, out_features=1024, bias=True)
  (output): Linear(in_features=1024, out_features=10, bias=True)
)

For model2:
MLP(
  (hidden_1): Linear(in_features=784, out_features=256, bias=True)
  (hidden_2): Linear(in_features=256, out_features=256, bias=True)
  (hidden_3): Linear(in_features=256, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```
Then you should train your network, just use one line:

```python
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

pretrained_model = train(model, criterion, optimizer)
```
Maybe you want to know the num of params, get it.

```python
print("num of params:", sum(param.numel() for param in pretrained_model.parameters()))
```
Or save the model:

```python
torch.save(pretrained_model.state_dict(), model_path)
```
Anyway, the most important thing is that modify the network and get the results of performance. Just create a function what you want to change in the network.Pruning? Quantification? Approximate compute? That is all ok.

```python
# verify the unchanged model
model = MLP()
model.load_state_dict(torch.load(model_path))
res1 = verify(model)

# verify the changed model, you can create functions to change the model just like "update_weight"!
res2 = []
for i in range(inject_times):
    model.load_state_dict(torch.load(model_path))
    res2.extend(verify(update_weight(model, path)))

```

Finally, have fun and good luck!

```bash
# This is a output in my changed network

model: MLP(
  (hidden_1): Linear(in_features=784, out_features=256, bias=True)
  (hidden_2): Linear(in_features=256, out_features=256, bias=True)
  (hidden_3): Linear(in_features=256, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
[1/3] Loss: 29.015525, Acc: 0.097875
Finish 1 epoch, Loss: 28.954814, Acc: 0.098550
[2/3] Loss: 28.851888, Acc: 0.100125
Finish 2 epoch, Loss: 28.827062, Acc: 0.106383
[3/3] Loss: 26.920954, Acc: 0.219875
Finish 3 epoch, Loss: 24.923297, Acc: 0.292733
num of params: 335114
current Function [my_test_train] run time is 36.03 s
current Function [run] run time is 11.96 s
```
## Feature

* Flexible: any layers and any layer size mlp network
* Decoupled: insert or remove different multiple function
* Simply: use one line to build your model and a function to modify your model
* Friendly: know less about python and pytorch to Architecturer

## Future
- [ ] Different network architecture,for example,CNN DNN RNN
- [ ] More DataSets
- [ ] More optimizations


 





