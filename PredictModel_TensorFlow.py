
# coding: utf-8

# In[1]:

import tensorflow as tf

#Categorical base columns
has_officers_comp = tf.contrib.layers.sparse_column_with_keys(column_name="has_officers_comp", keys=[0, 1])
has_existing_debt = tf.contrib.layers.sparse_column_with_keys(column_name="has_existing_debt", keys=[0, 1])

#Continuous base columns
amount = tf.contrib.layers.real_valued_column("amount")
amount_buckets = tf.contrib.layers.bucketized_column(amount, boundaries=[0, 50000,100000,150000,250000,300000,400000])
any_loan_coverage_ignoreRefi = tf.contrib.layers.real_valued_column("any_loan_coverage_ignoreRefi")
cash_for_debtPmts_official = tf.contrib.layers.real_valued_column("cash_for_debtPmts_official")
smartbiz_loan_coverage = tf.contrib.layers.real_valued_column("smartbiz_loan_coverage")
min_prin_score = tf.contrib.layers.real_valued_column("min_prin_score")
tx1_c__netProfitLoss = tf.contrib.layers.real_valued_column("tx1_c__netProfitLoss")
tx1_d__interestExpense = tf.contrib.layers.real_valued_column("tx1_d__interestExpense")
tx1_e__depreciation = tf.contrib.layers.real_valued_column("tx1_e__depreciation")
tx1_ga__annualSBLoanPmts = tf.contrib.layers.real_valued_column("tx1_ga__annualSBLoanPmts")
tx1_j__annualBusDebtPmts = tf.contrib.layers.real_valued_column("tx1_j__annualBusDebtPmts")
existing_loan_coverage_zerod = tf.contrib.layers.real_valued_column("existing_loan_coverage_zerod")


# In[2]:

wide_columns = [amount, any_loan_coverage_ignoreRefi, cash_for_debtPmts_official, smartbiz_loan_coverage, min_prin_score, tx1_c__netProfitLoss, tx1_d__interestExpense,tx1_e__depreciation, tx1_ga__annualSBLoanPmts,tx1_j__annualBusDebtPmts,existing_loan_coverage_zerod]
deep_columns = [amount, any_loan_coverage_ignoreRefi, cash_for_debtPmts_official, smartbiz_loan_coverage, min_prin_score, tx1_c__netProfitLoss, tx1_d__interestExpense,tx1_e__depreciation, tx1_ga__annualSBLoanPmts,tx1_j__annualBusDebtPmts,existing_loan_coverage_zerod]


# In[3]:

import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])


# In[4]:

import pandas as pd
df_train = pd.read_csv("final_data_train.csv")
df_test = pd.read_csv("final_data_test.csv")


# In[5]:

COLUMNS = ["has_officers_comp","has_existing_debt", "amount", "any_loan_coverage_ignoreRefi", "cash_for_debtPmts_official", "smartbiz_loan_coverage", "min_prin_score", "tx1_c__netProfitLoss", "tx1_d__interestExpense","tx1_e__depreciation", "tx1_ga__annualSBLoanPmts","tx1_j__annualBusDebtPmts","existing_loan_coverage_zerod"]
CATEGORICAL_COLUMNS = ["has_officers_comp","has_existing_debt"]
CONTINUOUS_COLUMNS = ["amount", "any_loan_coverage_ignoreRefi", "cash_for_debtPmts_official", "smartbiz_loan_coverage", "min_prin_score", "tx1_c__netProfitLoss", "tx1_d__interestExpense","tx1_e__depreciation", "tx1_ga__annualSBLoanPmts","tx1_j__annualBusDebtPmts","existing_loan_coverage_zerod"]


# In[6]:

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["performance_category"].apply(lambda x: "good" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["performance_category"].apply(lambda x: "good" in x)).astype(int)


# In[7]:

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = continuous_cols.copy() 
  feature_cols.update (categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)


# In[12]:

m.fit(input_fn=train_input_fn, steps=5000)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


# In[ ]:



