import pandas as pd
from sklearn.model_selection import train_test_split

tourism_df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

tourism_df.drop(columns=["CustomerID"], inplace=True)

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                        # Age of the customer
    'CityTier',                   # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'DurationOfPitch',            # Duration of the sales pitch delivered to the customer
    'NumberOfPersonVisiting',     # Total number of people accompanying the customer on the trip
    'NumberOfFollowups',          # Number of products the customer has with the bank
    'PreferredPropertyStar',      # Preferred hotel rating by the customer
    'NumberOfTrips',              # Average number of trips the customer takes annually.
    'Passport',                   # Whether the customer holds a valid passport (0: No, 1: Yes)
    'PitchSatisfactionScore',     # Score indicating the customer's satisfaction with the sales pitch
    'OwnCar',                     # Whether the customer owns a car (0: No, 1: Yes)
    'NumberOfChildrenVisiting',   # Number of children below age 5 accompanying the customer
    'MonthlyIncome'               # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',              # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation',                 # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender',                     # Gender of the customer (Male, Female)
    'ProductPitched',             # The type of product pitched to the customer
    'MaritalStatus',              # Marital status of the customer (Single, Married, Divorced)
    'Designation'                 # Customer's designation in their current organization
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_df[numeric_features + categorical_features]

# Define target variable
y = tourism_df[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("Data prepared: train/test splits written.")
