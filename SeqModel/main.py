# Main execution script
from functions import *

print(os.getcwd())
# cwd = os.chdir(os.path.join(os.getcwd(), "logda"))

File_xlsx = "data/integration_log_group_a.xlsx"

df = pd.read_excel(File_xlsx)
df = df.iloc[:, :-1] 

print(df.head())
# augment group_b at the end of it
File_xlsx = "data/integration_log_group_b.xlsx"


df_b = pd.read_excel(File_xlsx)
df_b = df_b.iloc[:, :-1]

df = pd.concat([df, df_b], axis=0)
print(df)
# read the scores
File_score_xlsx = ["data/Groupa_scores.xlsx", "data/Groupb_scores.xlsx"]

for i in range(2):
    # read the scores
    read_file = pd.read_excel(File_score_xlsx[i])

    # change the first column name to username
    read_file.rename(columns={"Participant ID's":'username', 'Problem 1 (Score out of 16)': 'score1',
                            'Problem 2 (Score out of 20)': 'score2', 'Problem 3 (Score out of 18)': 'score3' }, inplace=True)
    
    # use the username as the index
    read_file.set_index('username', inplace=True)
    # devide first column by 16, second column by 20, third column by 18
    read_file['score1'] = read_file['score1'].div(16)
    read_file['score2'] = read_file['score2'].div(20)
    read_file['score3'] = read_file['score3'].div(18)
    if i:
        read_file = read_file[:-4]
        score_df = pd.concat([score_df, read_file], axis=0)
    else:
        read_file = read_file[:-3]
        score_df = read_file.copy()

print(score_df)
# convert the score_df to a df series the value is the score.
# it should look like this:
# username probelm score
# a1        1       0.5
# a1        2       0.6
# a1        3       0.7
# a2        1       0.6


score_df = score_df.stack().reset_index()
score_df.columns = ['username', 'problem', 'score']
# replace score1, score2, score3 with 1, 2, 3
score_df['problem'] = score_df['problem'].str.replace('score', '').astype(int)
print(score_df)
# have a copy of the original df (run to reset)
if 'df_org' not in globals():
    df_org = df.copy()
else:
    df = df_org.copy()
# get the unique values of the column action
unique_actions = df.action.unique()
# peint unique actions with their counts
print(df['action'].value_counts())
# remove entries in df that are 'Save note','Auto-save log', 'Load default toolbox page', 'Load page', 'Run code error', 'Start reading sub-module', 'User login', 'User logout',  'Start problem', 'Submit problem'
df = df[~df['action'].isin(['Save note','Auto-save log', 'Load default toolbox page', 'Load page', 'Run code error', 'Start reading sub-module', 'User login', 'User logout',  'Start problem', 'Submit problem', 'Stop reading sub-module preview', 'Chatbot response'])]
# get the unique values of the column action
unique_actions = df.action.unique()
# peint unique actions with their counts
print(df['action'].value_counts())
df['space'].value_counts()
df['space'] = df['space'].replace('Problem 3', 3)
df['space'] = df['space'].replace('Problem 2', 2)
df['space'] = df['space'].replace('Problem 1', 1)
# if the space is not int, make it np.nan
df['space'] = pd.to_numeric(df['space'], errors='coerce')
# # if the space is not 1,2,3 change it to the value of the previous row
df['space'] = df['space'].fillna(method='ffill')
# change dtyle of space to Int64
df['space'] = df['space'].astype('Int64')
df['space'].value_counts()

# check for type(df.iloc[i, 2]) if is not datetime.time, then convert it to datetime.time

df = df.drop(columns=['Date and Time']) 
for i in range(len(df)):
    if type(df.iloc[i, 1]) != datetime.time:
        df.iloc[i, 1] = df.iloc[i, 1].time()
# rename second column to time
df.rename(columns={df.columns[1]: 'time'}, inplace=True)
df.rename(columns={df.columns[2]: 'problem'}, inplace=True)

# create a map for actions
# create a map for actions
action_map = {
    'Change answer': 'UA',
    'First answer': 'FA',
    'Paste answer': 'PA',
    'Request first hint': 'FH',
    'Request another hint': 'UH',
    'Respond to hint feedback': 'RH',
    'New answer explanation': 'FE',
    'Update answer explanation': 'UE',
    'Freeform code run': 'RF',
    'Run code': 'RC',
    'User request': 'B',
    'Update confidence': 'C',
    'Complete sub-module': 'M',
    'Streamlit interaction': 'S'
}

# map the actions
df['action'] = df['action'].map(action_map)
df

# in column time, compute the difference between the current row and the previous row in seconds, only if the username and probem are the same
# otherwise, put np.nan
df['duration']=np.nan
for i in range(1, len(df)):
    if df.iloc[i, 0] == df.iloc[i-1, 0] and df.iloc[i, 2] == df.iloc[i-1, 2]:
        df.iloc[i, 4] = (datetime.datetime.combine(datetime.datetime.today(), df.iloc[i, 1])
            - datetime.datetime.combine(datetime.datetime.today(), df.iloc[i-1, 1])).total_seconds()
# remove the time column
df = df.drop(columns=['time'])
df
# if there is a negative value in the duration column, replace it with np.nan
df['duration'] = df['duration'].apply(lambda x: np.nan if x < 0 else x)
# if it is greater than 1200, replace it with np.nan
df['duration'] = df['duration'].apply(lambda x: np.nan if x > 1200 else x)

# impute the missing values in duration with the mean of the duration in the same problem and username
df['duration'] = df['duration'].fillna(df.groupby(['username', 'problem'])['duration'].transform('mean'))
df
# describe the df 
df.describe()
# decode the duration colomn as follows:
# 0-1: '0'
# 1-5: '1'
# 5-10: '2'
# 10-15: '3'
# 15-20: '4'
# 20-30: ''5'
# 30-60: '6'
# 60-120: '7'
# 120-300: '8'
# 300-MAX: 'MAX'

df['duration'] = pd.cut(df['duration'], bins=[-0.1, 1, 5, 10, 15, 20, 30, 60, 120, 300, df['duration'].max()], labels=['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'TMAX'])
df

# for each username and problem, create a numpy array of the pair of actions, duration in the order they appear
# Then put the numpy array in a dictionary with the key as the username and problem
# example:
# (a1, 1): [UA, T0, FA, T1, PA, T2]
# (a1, 2): [UA, T0, FA, T1, PA, T2, FH, T3]
# (a2, 1): [UA, T0, FA, T1, PA, T2, FH, T3, UH, T4]

# create a dictionary
df_dict = {}
for i in range(len(df)):
    if (df.iloc[i, 0], df.iloc[i, 1]) in df_dict:
        df_dict[(df.iloc[i, 0], df.iloc[i, 1])].extend([df.iloc[i, 2], df.iloc[i, 3]])
    else:
        df_dict[(df.iloc[i, 0], df.iloc[i, 1])] = ['Q{}'.format(df.iloc[i, 1]), df.iloc[i, 2], df.iloc[i, 3]]

# # convert the list to numpy array
# for key in df_dict.keys():
#     df_dict[key] = np.array(df_dict[key])
     

print("number of sequence episode we have", len(df_dict.keys()))
print("number of scores we have ", len(score_df))
# for each key in the dictionary, find the score and append the values to a list (sequences) and the score to another list (scores)
# len(sequences) should be equal to len(scores)

sequences = []
scores = []
for key in df_dict.keys():
        sequences.append(df_dict[key])
        scores.append(score_df[(score_df['username'] == key[0]) & (score_df['problem'] == key[1])]['score'].values.tolist())

print(len(sequences), len(scores))
# if scores is empty, replace it with 0
scores = [[0] if len(x) == 0 else x for x in scores]
wandb.init(project="DaTu_prediction", entity="marslab", name = "bert-base-uncased")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)  # num_labels=1 for regression

new_tokens = ['UA','FA','PA','FH', 'UH', 'RH','FE', 'UE','RF', 'RC', 'B', 'C','M', 'S',
    'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'TMAX', 
    'Q1', 'Q2', 'Q3']

tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))  # Adjust model embedding size to include new tokens

# chceck if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Create the dataset and dataloader
dataset = DaTuDataset(sequences,  scores, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=None)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=None)

len(dataset)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# Optimizer and Learning Rate Scheduler
epochs = 16
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

train_val_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)
wandb.init(project="DaTu_prediction", entity="marslab", name = "distilbert-base-uncased")
new_tokens = ['UA','FA','PA','FH', 'UH', 'RH','FE', 'UE','RF', 'RC', 'B', 'C','M', 'S',
    'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'TMAX', 
    'Q1', 'Q2', 'Q3']
tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))  # Adjust model embedding size to include new tokens
#
# optional

model.classifier = nn.Sequential(
    nn.Dropout(0.3),  # Increase dropout
    nn.Linear(model.classifier.in_features, 1)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model.to(device)
# Create the dataset and dataloader
dataset = DaTuDataset(sequences,  scores, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=None)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=None)
epochs = 16
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
train_val_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)
wandb.init(project="DaTu_prediction", entity="marslab", name = "distilbert-Ciriculum")
# Create the dataset and dataloader
dataset = CurriculumDaTuDatasett(sequences,  scores, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=None)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=None)
epochs = 16
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
train_val_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)
wandb.init(project="DaTu_prediction", entity="marslab", name = "lora-Ciriculum-distilbert")
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=8, 
    lora_dropout=0.1, 
    target_modules="all-linear",
    bias="all"
)
model_peft = get_peft_model(model, peft_config)
model_peft.print_trainable_parameters()
epochs = 16
optimizer = AdamW(model_peft.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
train_val_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)
wandb.init(project="DaTu_prediction", entity="marslab", name = "augment-lora-Ciriculum-distilbert")
augmented_sequences = augment_sequence(sequences, max_changes=10, action_prob = 0.005, time_prob=0.03)
print(len(augmented_sequences), len(sequences))
# Create the dataset and dataloader
dataset = CurriculumDaTuDatasett(augmented_sequences,  scores, tokenizer)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, _ = random_split(dataset, [train_size, val_size])  # keep the validation the same

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=None)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=None)
epochs = 16
optimizer = AdamW(model_peft.parameters(), lr=5e-5)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
train_val_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)