from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def evaluate_models(models, X_test, y_test):

    print("\n")
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)

    best_model = None
    best_f1 = 0

    for name, model in models.items():

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n",name)
        print("-"*40)
        print("Accuracy :",accuracy)
        print("Precision:",precision)
        print("Recall   :",recall)
        print("F1 Score :",f1)

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test,y_pred))

        if f1>best_f1:
            best_f1=f1
            best_model=model

    return best_model