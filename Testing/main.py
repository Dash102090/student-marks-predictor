import tkinter as tk
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tkinter import ttk

window = tk.Tk()
window.geometry('500x500')

df = pd.read_csv('Student_Marks.csv')

Size = len(df)
Col_Names = list(df.columns)

treeview = ttk.Treeview(window,columns=(Col_Names), show='headings')

h_scrollbar = tk.Scrollbar(window, orient='horizontal', command=treeview.xview)
h_scrollbar.pack(side='bottom', fill='x')

v_scrollbar = tk.Scrollbar(window, orient='vertical', command=treeview.yview)
v_scrollbar.pack(side='right', fill='y')

treeview.config(xscrollcommand=h_scrollbar.set)
treeview.config(yscrollcommand=v_scrollbar.set)

treeview.pack(side='left', fill='both')

for i, names in enumerate(Col_Names):
    treeview.heading(names, text=names)


for i, rows in df.iterrows():
    treeview.insert('', 'end', values=list(rows))

Diabetes = load_diabetes()

fd = pd.DataFrame(data=Diabetes.data, columns=Diabetes.feature_names)


X = df.drop('Marks', axis=1)
y = df['Marks']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print(f'R2 = {round(r2, 4)}')

mse = mean_squared_error(y_test, y_pred)
print(f'Mean_Squared_Mean = {round(mse, 4)}')


print(f'Coefficient = {model.coef_}\nIntercept = {model.intercept_}')

Scatter = plt.scatter(y_test, y_pred, color='#6C60C4')
Plot = plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#FF9317')

plt.title('Predicted Marks')
plt.xlabel('Time Studied')
plt.ylabel('Marks')
plt.show()

window.mainloop()
