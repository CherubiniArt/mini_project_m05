from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class Analysis():

    def __init__(self, y_pred_rf, target, mean_target, std_target, flag):
        self.y_pred_rf_train =y_pred_rf[0]
        self.y_pred_rf_cv = y_pred_rf[1]
        self.y_pred_rf_test = y_pred_rf[2]

        self.y_train = target[0]
        self.y_cv = target[1]
        self.y_test = target[2]

        self.mean = mean_target
        self.std = std_target

        self.flag = flag

    def __call__(self):
        # Compute r2:
        r2_train = r2_score(self.y_train, self.y_pred_rf_train)
        r2_cv = r2_score(self.y_cv, self.y_pred_rf_cv)
        r2_test = r2_score(self.y_test, self.y_pred_rf_test)

        print("r2 train =", r2_train)
        print("r2 cv =", r2_cv)
        print("r2 test =", r2_test)

        plt.bar(["train", "cv", "test"], [r2_train, r2_cv, r2_test])
        plt.ylabel("r2")
        title = " Random Forest" if self.flag=="RF" else " Decision Tree Regression"
        plt.title("Performance of the" + title  + " algorithm")
        plt.show()
