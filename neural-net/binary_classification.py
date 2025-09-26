
import torch
import torch.nn as nn
import torch.optim as optim

# lets generate random data
X = torch.rand(1000,2) # 1000 samples, 2 random numbers between 0 and 1, per entry
# this is going to be our Tensor (multi dimensional array) which represents data and features IRL.
print(f"x is {X.shape}")


# if sum > 1 , then we wanna label it as 1 else label it as 0
y = (X[:,0] + X[:,1] > 1).float().unsqueeze(1) # our labelling function/array
# for each sample, it adds two features: if sum is > 1, its true, else its false
# .float() converts True/False into 1.0 or 0.0
# .unsqueeze(1) converts the shape [1000] to [1000,1] so that we can match the training shape, cuz remember, our x was [1000,2], but based on our y, its gotta end up being [1000,1] cuz its either true or false per entry, cuz we just labelled it

print(f"y is {y.shape}")

# Lets define a NN
class sguneNN(nn.Module): #putting nn.Module there means we're inheriting stuff in
    def __init__(self): #this is our constructor and runs as soon as we create a class object
        super().__init__() # inits the nn.Module class, so PyTorch can track layers and params

        self.fc1 = nn.Linear(2,4) # layer 1 (input layer)
        #Input is 2 features (FROM X matrix), but outputs 4 neurons
        #These 4 will server as our "hidden layer", and is basically mapping weights from 2D space to 4D space

        self.relu = nn.ReLU()
        # adds NON-LINEARITY so that the model can learn complex patterns
        # we're using ReLU cuz its fast and helps avoid vanishing gradients
        # need more info


        self.fc2 = nn.Linear(4,1) # output layer
        # Input is 4 features (from the previous layer), and output is a single probability value

        self.sigmoid = nn.Sigmoid()
        # this helps squash the output of fc2 to give us a probability value between 0 and 1, which helps with binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    
model = sguneNN()
print(f"Model object created: {model}")

# Lets define loss and optimizer
criterion = nn.BCELoss()
# BCELoss is Binary Cross-Entropy Loss, and is used for binary classification problems. Helps measure how well the predicted probabilities match the actual labels

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)
# Adam is just a popular optimizer that mods the learning rate during training phase.
# model.parameters is just giving the weights and biases to the optimizer, and lr defines how BIG of a step we're taking during each epoch


def runIT(sample, runs, X, optimizer, criterion, model):
    finalLoss = 0
    # Lets train
    for epoch in range(runs):
        optimizer.zero_grad() #clear the gradients from the previous pass. We don't want previous runs to affect our current run
        outputs = model(X) # time to feed our model with our X array
        loss = criterion(outputs, y) #compares our predicted values to the labels from y. We're basically doing this to calculate how far off our values are so we can tweak it on the next run.
        loss.backward() # do backpropagation, ie, compute gradients of loss w.r.t model parameters
        optimizer.step() # Applies gradients to model's params
        # print(
        #     f"""
        #         outputs: {outputs},
        #         loss : {loss},
        #         backpropagation: DONE,
        #         optimizer: DONE
        #     """
        # )
        print(f"Epoch: {epoch+1}, Loss:{loss.item():.4f}")
        finalLoss = loss.item()
    with torch.no_grad():
        prediction = model(sample)
        print(f"OG: {sample}, Prediction: {prediction.item()} \nwhich means, I'm {prediction.item()*100} % confident")
        predicted_class = 1 if prediction.item() > 0.5 else 0
        print(f"Sample Sum of {sample} is : {sample[:,0]+sample[:,1]} and based on that, it feels like its {predicted_class}")


    print(f"Final Loss after {runs} runs is {finalLoss:.4f}")

# Testing time
sample1 = torch.tensor([[0.4, 0.3]]) # this adds up to 0.7, which is < 1, so we expect output prediction of 0

runIT(sample1, 100, X, optimizer, criterion, model)

sample2 = torch.tensor([[0.5, 0.6]]) # this adds up to 1.1, which is > 1, so we expect output prediction of 1

runIT(sample2, 100, X, optimizer, criterion, model)




