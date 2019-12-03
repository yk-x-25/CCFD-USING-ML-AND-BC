# Credit card Fraud Detection ML Algorithm 
Logistic regression is one of the major classification Algorithm that is widely used. Logistic regression is a statistical model that makes use of Sigmoid function as the learning curve.
# Understanding the Dataset
The dataset used is an opensource from kaggle Here's the link -> https://www.kaggle.com/mlg-ulb/creditcardfraud. 
* The dataset used has about 30+ fields or parameters that we'll be calling in the ML World out of which most of them are out of scope to this prototype model. However, We still don't know about exactly what those fields are (probably due to privacy).
* So, We Start with an Understanding that those parameters have a very little impact on the whole model and hence it would matter a lot if we omit those parameters  
* However, we are left with 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# Getting Started
  ## Step #1 Getting hands Dirtier 
* ### Understanding True Positives, False positives, True Negatives, False Negatives
     Let's take an Example that Mr 'X' has Cancer and we are to make a ML Model to detect whether or not he has cancer.
     * #### True Postive:
          * Our Model is so well trained that it has successfully detected that mr 'X' has cancer .Then, it is ``` True Positive. ```
     * #### False Postive:
          * If our model fails to identify the cancer though Mr 'X' has it. Then, It's ``` False positive .```
     
     Now, Mr 'Y' is a Normal Healthy person comes for a normal check up 
     * #### True Negative:
          * Our Model is so well trained that it has successfully detected that mr 'Y' doesn't have cancer .Then, it is ``` True Negative. ```
     * #### False Negative:
          * If our model Wrongly detects cancer for a healthy man Mr. 'Y'. Then, It's ``` False Negative .```

Before getting started with what this model does, Let's get to know some important stuff before landing into the model Conclusion.


* ### Accuracy: 
     * We've heard many a times ,the phrase "How close it is?". Well, the phrase can mean different depending on the refernce we are comparing to. If we're comparing to the actual target, it becomes "Accuracy" .But,If we're comparing to the previous encounter,that is "Precision". 
     * So, In Machine learning, Accuracy is how close the predicted value of your model is to the actual outcome.It's obvious that we can have accuracy only for "Supervised Machine Learning Algorithms"
* ### Precision:
     * As I said before, We only focus on the relevent points that are correct and we leave behind the wrong ones. The Correct one's directly mean that we're looking for true positives. And we are comparing relative to all the data points that out model computes!.
      
   
    <img src=https://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%20&plus;%20FP%7D%20%7D>
