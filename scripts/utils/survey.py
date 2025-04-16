import os
import pandas as pd

def preprocess_survey(survey_df):
    survey_df = survey_df.iloc[2:]
    new_survey_df = survey_df[[
        f'Q{i}' for i in range(3, 8)
        ]+['id',]].copy()
    
    # there are 7 subquestions for q1
    for qid in [f'Q1.{i+1}' for i in range(7)]:
        if qid in survey_df.columns:
            new_survey_df[qid] = survey_df[qid]

    # some have two sub questions for q5
    for qid in [f'Q5.{i+1}' for i in [2, 3]]:
        if qid in survey_df.columns:
            new_survey_df[qid] = survey_df[qid]
        
    survey_df = new_survey_df
    survey_df.reset_index(drop=True, inplace=True)
    survey_df.set_index('id', inplace=True)
    survey_df.index = survey_df.index.astype(int) # .astype(str)
    survey_df = survey_df.apply(pd.to_numeric)
    return survey_df

def load_all_surveys(dpaths):
    surveys = []
    for dpath in dpaths:
        survey_path = os.path.join(dpath, 'survey.csv')
        print(survey_path)
        survey = pd.read_csv(survey_path)
        survey = preprocess_survey(survey)
        surveys.append(survey)
    surveys = pd.concat(surveys)
    return surveys