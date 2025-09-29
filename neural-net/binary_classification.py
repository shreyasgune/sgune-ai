
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# lets generate random data
X = torch.rand(100,2) # 1000 samples, 2 random numbers between 0 and 1, per entry
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
# model.parameters is just giving the weights and biases to the optimizer, and lr defines how BIG of a 
#step we're taking during each epoch


## Some viz stuff
# Remember:
## Red regions = model predicts class 1 (sum of x+y > 1)
## Blue regions = class 0 (sum ≤ 1)
# 🔵 Blue background	Model thinks: class 0 (low probability)
# 🔴 Red background	Model thinks: class 1 (high probability)
# ⚫ Dashed black line	The true boundary: x + y = 1
# 🔴 Red points	Actual data with label 1
# 🔵 Blue points	Actual data with label 0
# 🎨 Colorbar	Probability values (0 to 1)
def plot_decision_boundary(model, X, y, sample=None):
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid).reshape(xx.shape).numpy()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    plt.colorbar(contour, label="Model's Probability of Class 1")

    # Plot ground truth boundary: x + y = 1
    x_line = np.linspace(0, 1, 100)
    y_line = 1 - x_line
    plt.plot(x_line, y_line, 'k--', label="Ground Truth Boundary (x + y = 1)")

    # Plot training data
    X_np = X.numpy()
    y_np = y.numpy().ravel()
    plt.scatter(X_np[y_np == 0][:, 0], X_np[y_np == 0][:, 1],
                color='blue', edgecolor='k', s=30, label='Class 0')
    plt.scatter(X_np[y_np == 1][:, 0], X_np[y_np == 1][:, 1],
                color='red', edgecolor='k', s=30, label='Class 1')

    # Highlight the sample input
    if sample is not None:
        sample_np = sample.numpy()
        pred = model(sample).item()
        plt.scatter(sample_np[0, 0], sample_np[0, 1],
                    color='limegreen', edgecolor='black', s=150,
                    label=f'Sample (Pred: {pred:.2f})', zorder=5)

        #sample coordinates
        plt.annotate(f"({sample_np[0,0]:.2f}, {sample_np[0,1]:.2f})",
                     (sample_np[0, 0] + 0.02, sample_np[0, 1] + 0.02),
                     color='green', fontsize=10)

    plt.title("Decision Boundary with Training Data and Sample")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Looking at this graph, we can see at what iteration, is our model likely to converge
def plotLoss(runs,loss):
    # create x-axis data for runs
    iterations = np.arange(1,runs+1)

    if len(iterations) != len(loss):
        print(f"Error: Iterations ({len(iterations)}) does not match length {len(loss)}")
        return
    
    plt.plot(iterations, loss)
    plt.title("Loss Plot")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


def runIT(sample, runs, X, y, optimizer, criterion, model):
    finalLoss = []
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
        # print(f"Epoch: {epoch+1}, Loss:{loss.item():.4f}")
        finalLoss.append(loss.item())
    print("\nModel Weights and Biases:")
    print("fc1 weights:\n", model.fc1.weight.data)
    print("fc1 biases:\n", model.fc1.bias.data)
    print("fc2 weights:\n", model.fc2.weight.data)
    print("fc2 biases:\n", model.fc2.bias.data)
    # print(f"Losses:",finalLoss)
    plotLoss(runs,finalLoss)

    with torch.no_grad():
        prediction = model(sample)
        print(f"OG: {sample}, Prediction: {prediction.item()} \nwhich means, I'm {prediction.item()*100} % confident")
        predicted_class = 1 if prediction.item() > 0.5 else 0
        print(f"Sample Sum of {sample} is : {sample[:,0]+sample[:,1]} and based on that, it feels like its {predicted_class}")


    print(f"Final Loss after {runs} runs is {finalLoss[len(finalLoss)-1]:.4f}")
    plot_decision_boundary(model,X,y, sample)

# Testing time
sample1 = torch.tensor([[0.4, 0.3]]) # this adds up to 0.7, which is < 1, so we expect output prediction of 0

runIT(sample1, 1000, X, y, optimizer, criterion, model)

sample2 = torch.tensor([[0.5, 0.6]]) # this adds up to 1.1, which is > 1, so we expect output prediction of 1

runIT(sample2, 1000, X, y, optimizer, criterion, model)




