import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.uic import loadUiType
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QGraphicsScene
import joblib
file_path = r"D:\PYthon Projects\machine learning project\train.csv"

FORM_CLASS, _ = loadUiType(r"D:\PYthon Projects\machine learning project\GUI_ML.ui")


class TitanicModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df_raw = None
        self.df_dropped = None
        self.df_filled = None
        self.df_encoded = None
        self.df_engineered = None
        self.df_scaled = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.models = []
        self.model_names = []




            
    def load_and_process_data(self):
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "اختر ملف CSV", "", "CSV Files (*.csv);;All Files (*)")
            self.file_path=file_path
            self.df_raw = pd.read_csv(file_path)

            self.show_message("The file data has been successfully loaded, ready for processing ✅")

            
    def process_data(self):
        self.drop_columns()
        self.fill_missing()
        self.IQR()
        self.encode_features()
        self.engineer_features()
        self.scale_features()
        self.split_data()
        self.train_models()
        
    # _________________________________________________________________
    def drop_columns1(self):
        self.df_dropped1 = self.df_raw.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        self.show_message("Unnecessary columns have been removed to clean the data.")

        return self.df_dropped1
    
    def fill_missing1(self):
        df = self.df_dropped1.copy()
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        self.df_filled1 = df
        self.show_message("Successfully filled missing values in 'Age' with median and in 'Embarked' with mode.")

        return self.df_filled1

        
    def encode_features1(self):
        df_encoded = self.df_iqr.copy()
    
        df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
        embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked', drop_first=False)
        embarked_dummies = embarked_dummies.astype(int)  
        df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
        df_encoded.drop('Embarked', axis=1, inplace=True)
        
        self.show_message("Converted the 'Sex' column: Male = 0, Female = 1")


        
        self.show_message("Port data has been converted to numerical values suitable for the model.")


    
        self.df_encoded1 = df_encoded
        return self.df_encoded1    

    def engineer_features1(self):
        df = self.df_encoded1.copy()
        df['FamilySize'] = df['SibSp'] + df['Parch']
        df.drop(columns=['SibSp', 'Parch'], inplace=True)
        self.df_engineered1 = df
        self.show_message("Family size has been calculated and old columns have been removed.")


        return self.df_engineered1
    def scale_features1(self):
        df = self.df_engineered.copy()
        scaler = StandardScaler()
        df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])
        self.df_scaled1 = df
        self.show_message("Age, fare, and family size values have been adjusted to suit the model.")

        return self.df_scaled1
    
    
    # _______________________________________________________________-
    def drop_columns(self):
        self.df_dropped = self.df_raw.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        
    def fill_missing(self):
        df = self.df_dropped.copy()
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        self.df_filled = df
        
    def IQR(self):
        df_iqr=self.df_filled.copy()
        numeric_cols = df_iqr.select_dtypes(include=['number']).columns
        Q1 = df_iqr[numeric_cols].quantile(0.25)
        Q3 = df_iqr[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df_iqr = df_iqr[~((df_iqr[numeric_cols] < (Q1 - 1.5 * IQR)) | (df_iqr[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        self.df_iqr=df_iqr
        
    def encode_features(self):
        df_encoded = self.df_iqr.copy()
    
        df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

        embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked', drop_first=False)
        embarked_dummies = embarked_dummies.astype(int)  
        
        df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
        df_encoded.drop('Embarked', axis=1, inplace=True)
        
        self.df_encoded=df_encoded
         

            
    def engineer_features(self):
        df = self.df_encoded.copy()
        df['FamilySize'] = df['SibSp'] + df['Parch']
        df.drop(columns=['SibSp', 'Parch'], inplace=True)
        self.df_engineered = df
       
    def scale_features(self):
        df = self.df_engineered.copy()
        scaler = StandardScaler()
        df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])
        self.df_scaled = df
        self.scaler = scaler  
        
    def describe(self):
        return self.df_raw.describe()
    
    def show_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Information")
        msg.exec_()
        
    # --------------------------------------------------------------
    def split_data(self):
        x = self.df_scaled.drop('Survived', axis=1)
        y = self.df_scaled['Survived']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    def train_models(self):
             self.model1 = DecisionTreeClassifier(criterion='entropy', max_depth=3).fit(self.x_train, self.y_train)
             self.model2 = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000).fit(self.x_train, self.y_train)
             self.model3 = SVC(kernel='linear', C=1.0, random_state=42).fit(self.x_train, self.y_train)
             self.model4 = KNeighborsClassifier(n_neighbors=5).fit(self.x_train, self.y_train)
             self.model5 = GaussianNB().fit(self.x_train, self.y_train)
             self.model6 = RandomForestClassifier(n_estimators=100, random_state=42).fit(self.x_train, self.y_train)



    def evaluate(self,model,model_name):
        y_pred = model.predict(self.x_test)
        accuracy = round(accuracy_score(self.y_test, y_pred), 2)
        precision = round(precision_score(self.y_test, y_pred), 2)
        recall = round(recall_score(self.y_test, y_pred), 2)
        f1 = round(f1_score(self.y_test, y_pred), 2)

        self.metrics_df = pd.DataFrame([[
            model_name, accuracy, precision, recall, f1
        ]], columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]).T
        
        return self.metrics_df


    def evaluate_roc_auc(self, model, model_name):
        auc_score = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(self.x_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc_score = round(roc_auc_score(self.y_test, y_prob), 2)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(self.x_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            auc_score = round(roc_auc_score(self.y_test, y_score), 2)
        else:
            print(f"{model_name} does not support probability output. Skipping ROC curve.")
            return pd.DataFrame([[model_name, "N/A"]], columns=["Model", "AUC"]).T
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc_score}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')  
        ax.set_title(f'ROC Curve: {model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
    
        canvas = FigureCanvas(fig)
    
        scene = QGraphicsScene(self)
        scene.addWidget(canvas)
    
        self.graphicsView_3.setScene(scene)
        
        canvas.draw()
        
        return
    
    
    def draw_confusion_matrices(self, model, name):
        
            y_pred = model.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
    
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
    
            canvas = FigureCanvas(fig)
    
            scene = QGraphicsScene(self)
            scene.addWidget(canvas)
    
            self.graphicsView_4.setScene(scene)
    
            canvas.draw()
    
    

    
FORM_CLASS, _ = loadUiType(r"D:\PYthon Projects\machine learning project\GUI_ML.ui")


class PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        """إرجاع عدد الصفوف في الـ DataFrame"""
        return self._df.shape[0]

    def columnCount(self, parent=None):
        """إرجاع عدد الأعمدة في الـ DataFrame"""
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """إرجاع البيانات من الـ DataFrame بناءً على الفهرس"""
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.4f}"  
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """إرجاع بيانات الرأس للعرض"""
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal: 
                return str(self._df.columns[section])
            elif orientation == Qt.Vertical:  
                return str(self._df.index[section])
        return None
    
    
class MainApp(QMainWindow, FORM_CLASS, TitanicModel):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        TitanicModel.__init__(self, file_path)
        # ____________________________________________
  
        self.tabWidget.tabBar().hide()
        self.tabWidget_2.tabBar().hide()
        self.tabWidget_3.tabBar().hide()
        self.tabWidget_4.tabBar().hide()      
        # ____________________________________________

        
        self.pushButton.clicked.connect(self.load_and_process_data)
        self.pushButton_2.clicked.connect(self.process_data)
        
        self.pushButton_2.clicked.connect(lambda: self.go_to_page(1))
        self.pushButton_7.clicked.connect(lambda: self.go_to_page(0))
        self.pushButton_8.clicked.connect(lambda: self.go_to_page(2))
        self.pushButton_9.clicked.connect(lambda: self.go_to_page(1))
        self.pushButton_10.clicked.connect(lambda: self.go_to_page(3))
        self.pushButton_16.clicked.connect(lambda: self.go_to_page(2))
        self.pushButton_17.clicked.connect(lambda: self.go_to_page(4))
        self.pushButton_18.clicked.connect(lambda: self.go_to_page2(1))
        self.pushButton_20.clicked.connect(lambda: self.go_to_page2(2))
        self.pushButton_28.clicked.connect(lambda: self.go_to_page(3))
        self.pushButton_29.clicked.connect(lambda: self.go_to_page(5))
        

        self.pushButton_30.clicked.connect(lambda: self.go_to_page(4))
        
        self.pushButton_3.clicked.connect(lambda: self.view_data(self.df_raw)  )
        self.pushButton_3.clicked.connect(lambda: self.switch_to_inner_tab(1, 0))

        self.pushButton_4.clicked.connect(lambda:self.view_data3(self.describe))
        self.pushButton_4.clicked.connect(lambda: self.switch_to_inner_tab(1, 1))
        
        # ________________________________________________________________________________
        self.pushButton_22.clicked.connect(lambda:self.view_data4(self.evaluate(self.model1,"Decision Tree")))
        self.pushButton_22.clicked.connect(lambda: self.switch_to_inner_tab2(4, 1))
        
        self.pushButton_23.clicked.connect(lambda:self.view_data4(self.evaluate(self.model2,"Logistic Regression")))
        self.pushButton_23.clicked.connect(lambda: self.switch_to_inner_tab2(4, 1))
        
        self.pushButton_24.clicked.connect(lambda:self.view_data4(self.evaluate(self.model3,"SVM")))
        self.pushButton_24.clicked.connect(lambda: self.switch_to_inner_tab2(4, 1)) 
        
        self.pushButton_27.clicked.connect(lambda:self.view_data4(self.evaluate(self.model4,"KNN")))
        self.pushButton_27.clicked.connect(lambda: self.switch_to_inner_tab2(4, 1))

        self.pushButton_25.clicked.connect(lambda:self.view_data4(self.evaluate(self.model5,"Gaussian Naive Bayes")))
        self.pushButton_25.clicked.connect(lambda: self.switch_to_inner_tab2(4, 1))
        
        self.pushButton_26.clicked.connect(lambda:self.view_data4(self.evaluate(self.model6,"Random Forest")))
        self.pushButton_26.clicked.connect(lambda: self.switch_to_inner_tab2(4, 1))   
            
            # ____________________________________________________________________________--
        self.pushButton_22.clicked.connect(lambda:self.evaluate_roc_auc(self.model1,"Decision Tree"))
    
        self.pushButton_23.clicked.connect(lambda:self.evaluate_roc_auc(self.model2,"Logistic Regression"))
    
        self.pushButton_24.clicked.connect(lambda:self.evaluate_roc_auc(self.model3,"SVM"))
    
        self.pushButton_27.clicked.connect(lambda:self.evaluate_roc_auc(self.model4,"KNN"))
        
        self.pushButton_25.clicked.connect(lambda:self.evaluate_roc_auc(self.model5,"Gaussian Naive Bayes"))
    
        self.pushButton_26.clicked.connect(lambda:self.evaluate_roc_auc(self.model6,"Random Forest"))
            
            
            # _______________________________________________________________________________
        self.pushButton_22.clicked.connect(lambda:self.draw_confusion_matrices(self.model1,"Decision Tree"))
    
        self.pushButton_23.clicked.connect(lambda:self.draw_confusion_matrices(self.model2,"Logistic Regression"))
    
        self.pushButton_24.clicked.connect(lambda:self.draw_confusion_matrices(self.model3,"SVM"))
    
        self.pushButton_27.clicked.connect(lambda:self.draw_confusion_matrices(self.model4,"KNN"))
        
        self.pushButton_25.clicked.connect(lambda:self.draw_confusion_matrices(self.model5,"Gaussian Naive Bayes"))
    
        self.pushButton_26.clicked.connect(lambda:self.draw_confusion_matrices(self.model6,"Random Forest"))
        self.predict_button.clicked.connect(self.predict_survival)    
        # ____________________________________________________________________________________
        self.pushButton_5.clicked.connect(lambda: self.switch_to_inner_tab(1, 2))


     
        self.pushButton_6.clicked.connect(lambda: self.switch_to_inner_tab(1, 3))

        
        self.pushButton_11.clicked.connect(lambda: self.view_data2(self.drop_columns1))
        self.pushButton_12.clicked.connect(lambda: self.view_data2(self.fill_missing1))
        self.pushButton_13.clicked.connect(lambda: self.view_data2(self.encode_features1))
        self.pushButton_14.clicked.connect(lambda: self.view_data2(self.engineer_features1))
        self.pushButton_15.clicked.connect(lambda: self.view_data2(self.scale_features1))
        self.type_relationship.currentIndexChanged.connect(self.display_selected_graph)
        

        # _______________________________الدوال الاساسيه _______________________________________________
    def go_to_page(self, number_page):
        self.tabWidget.setCurrentIndex(number_page)
    def go_to_page2(self, number_page):
        self.tabWidget_3.setCurrentIndex(number_page)


    def switch_to_inner_tab(self, indexin, indexout):
        self.tabWidget.setCurrentIndex(indexin)
        self.tabWidget_2.setCurrentIndex(indexout)    
    

    def switch_to_inner_tab2(self, indexin, indexout):
        self.tabWidget.setCurrentIndex(indexin)
        self.tabWidget_4.setCurrentIndex(indexout)    

    
    
    def view_data(self,function):
        model = PandasModel(function)
        self.tableView.setModel(model)  
        self.tableView.setAlternatingRowColors(True)

    def view_data2(self,function):
        df_result = function() 
        model = PandasModel(df_result)
        self.tableView_2.setModel(model)
        self.tableView_2.setAlternatingRowColors(True)
        
    def view_data3(self,function):
        df_result = function() 
        model = PandasModel(df_result)
        self.tableView_3.setModel(model)  
        self.tableView_3.setAlternatingRowColors(True)
        
    def view_data4(self,function):
        model = PandasModel(function)
        self.tableView_4.setModel(model)  
        self.tableView_4.setAlternatingRowColors(True)
    
    def view_data5(self,function):
        model = PandasModel(function)
        self.graphicsView_3.setModel(model)  
        self.graphicsView_3.setAlternatingRowColors(True)
    

    def display_selected_graph(self):
        selected = self.type_relationship.currentText()

        fig, ax = plt.subplots(figsize=(10, 6))

        if selected == "Box Plots: Age and Fare vs Survival":
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.boxplot(data=self.df_iqr, x='Survived', y='Age', ax=axes[0], color='lightblue')
            axes[0].set_title('Age Distribution by Survival')
            sns.boxplot(data=self.df_iqr, x='Survived', y='Fare', ax=axes[1], color='lightgreen')
            plt.tight_layout()
            axes[1].set_title('Fare Distribution by Survival')
            
       
                  
        if selected == "Distributions of Important Features (Before IQR Outlier Removal)":
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
                fig.suptitle('Distributions of Important Features Before Outliers IQR', fontsize=16)
        
                features = ['Age', 'Fare', 'Pclass']
                for ax, feature in zip(axes.flatten(), features):
                    ax.hist(self.df_raw[feature], bins=40, edgecolor='white')
                    ax.set_title(f'{feature} Distribution')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Count')
        
                plt.tight_layout(rect=[0, 0.02, 1, 0.88])            
                
        
        elif selected == "Survival Distribution by Sex":
            sns.countplot(data=self.df_engineered, x='Sex', hue='Survived', palette='magma', ax=ax)
            ax.set_title('Survival Distribution by Sex')
            ax.set_xlabel('Sex')
            ax.set_ylabel('Number of Passengers')
            ax.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
            
        elif selected == "Survival Distribution by Passenger Class":
            sns.countplot(data=self.df_engineered, x='Pclass', hue='Survived', palette='plasma', ax=ax)
            ax.set_title('Survival Distribution by Passenger Class')
            ax.set_xlabel('Passenger Class')
            ax.set_ylabel('Number of Passengers')
            ax.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
            
        elif selected == "Passenger Age Distribution":
            sns.histplot(self.df_engineered['Age'], kde=True, bins=30, color='skyblue', ax=ax)
            ax.set_title('Passenger Age Distribution')
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            
        
        elif selected == "Fare Distribution":
           fig, ax = plt.subplots(1, 2, figsize=(15, 6))

           sns.histplot(self.df_engineered['Fare'], kde=False, bins=50, ax=ax[0], color='lightgreen')
           ax[0].set_title('Fare Distribution (Full Range)')
           ax[0].set_xlabel('Fare')
           ax[0].set_ylabel('Frequency')
           
           sns.histplot(self.df_engineered[self.df_engineered['Fare'] < 100]['Fare'], kde=True, bins=40, ax=ax[1], color='lightcoral')
           ax[1].set_title('Fare Distribution (Fare < $100)')
           ax[1].set_xlabel('Fare')
           ax[1].set_ylabel('Frequency')
           
           plt.tight_layout()
           
           
        elif selected == "Fare Distribution by Survival Status":
        
           sns.boxplot(x='Survived', y='Fare', data=self.df_engineered, palette='winter', ax=ax)
           ax.set_title('Fare Distribution by Survival Status')
           ax.set_xticks([0, 1])
           ax.set_xticklabels(['Did Not Survive', 'Survived'])
           ax.set_xlabel('Outcome')
           ax.set_ylabel('Fare')
           ax.set_ylim(0, 300)  
           
           
        elif selected == "Distribution of Family Size (SibSp + Parch)":
            sns.countplot(x='FamilySize', data=self.df_engineered, palette='cubehelix', ax=ax)
            ax.set_title('Distribution of Family Size (SibSp + Parch)')
            ax.set_xlabel('Number of Family Members Aboard')
            ax.set_ylabel('Number of Passengers')
        
        elif selected == "Survival Rate by Family Size":
            
            sns.pointplot(x='FamilySize', y='Survived', data=self.df_engineered, errorbar=None, color='purple', ax=ax)
            ax.set_title('Survival Rate by Family Size')
            ax.set_xlabel('Family Size')
            ax.set_ylabel('Survival Rate')
            ax.grid(True, axis='y')
            
        
        elif selected == "Age vs Fare (Colored by Survival)":
            sns.scatterplot(data=self.df_engineered, x='Age', y='Fare', hue='Survived', palette='seismic', alpha=0.7, ax=ax)
            ax.set_title('Age vs Fare (Colored by Survival)')
            ax.set_xlabel('Age')
            ax.set_ylabel('Fare')
            ax.set_ylim(0, 300)  
            ax.legend(title='Survived', labels=['Did not Survive', 'Survived'])
            
        
        elif selected == "Age Distribution by Passenger Class":
            sns.violinplot(x='Pclass', y='Age', data=self.df_engineered, palette='Set3', ax=ax)
            ax.set_title('Age Distribution by Passenger Class')
            ax.set_xlabel('Passenger Class')
            ax.set_ylabel('Age')

            
  
        
        canvas = FigureCanvas(fig)
        scene = QGraphicsScene()
        scene.addWidget(canvas)
        self.graphicsView_2.setScene(scene)
        
        
        
    def predict_survival(self):
        try:
            age = float(self.age_input.text())
            sex = int(self.sex_input.currentText())
            fare = float(self.fare_input.text())
            pclass = int(self.pclass_input.currentText())
            familysize = int(self.family_input.value())  
            embarked_letter = self.embarked_input.currentText()  

            embarked_c = 1 if embarked_letter == 'C' else 0
            embarked_q = 1 if embarked_letter == 'Q' else 0
            embarked_s = 1 if embarked_letter == 'S' else 0

            numeric_df = pd.DataFrame([[age, fare, familysize]], columns=['Age', 'Fare', 'FamilySize'])
            numeric_scaled = self.scaler.transform(numeric_df)
            age_scaled, fare_scaled, familysize_scaled = numeric_scaled[0]

            input_features = [pclass, sex, age_scaled, fare_scaled, embarked_c, embarked_q, embarked_s, familysize_scaled]
            columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'FamilySize']
            input_df = pd.DataFrame([input_features], columns=columns)

            prediction = self.model6.predict(input_df)[0]

            if prediction == 1:
                QMessageBox.information(self, "Result", "Congratulations! The person survived the sinking ✅")
            else:
                QMessageBox.information(self, "Result", "Unfortunately, the person did not survive ❌")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    # _________________________________________________________________________________
    
def main():
     app = QApplication(sys.argv)
     window = MainApp()
     window.show()
     sys.exit(app.exec_())


if __name__ == '__main__':
    main()
