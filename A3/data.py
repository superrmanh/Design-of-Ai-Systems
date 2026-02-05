import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Creating a load Function for the Data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Loading in the Data
Beijing = load_csv('Cities/Beijing_labeled.csv')
Guangzhou = load_csv('Cities/Guangzhou_labeled.csv')
Shanghai = load_csv('Cities/Shanghai_labeled.csv')
Shenyang = load_csv('Cities/Shenyang_labeled.csv')

# Creating A City Column for the Cities
train_data = pd.concat([Beijing,Shenyang])
test_data  = pd.concat([Shanghai,Guangzhou])

# Seperating the Features and Target
X = train_data.drop(columns = 'PM_HIGH')
Y = train_data['PM_HIGH']

# Splitting the Data 80/20
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size =0.2, random_state=42)


