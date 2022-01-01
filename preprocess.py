import pandas as pd

data = pd.read_csv("dating-full.csv",nrows=6500)


def discretize(data,bins):

    excludedColumns = ['gender','samerace','decision']

    rangeDictionary = { 'age' : [18,58],
                        'age_o' : [18,58],
                        'importance_same_race' : [0,10],
                        'importance_same_religion' : [0,10],
                        'pref_o_attractive' : [0,1],
                        'pref_o_sincere' : [0,1],
                        'pref_o_intelligence' : [0,1],
                        'pref_o_funny' : [0,1],
                        'pref_o_ambitious' : [0,1],
                        'pref_o_shared_interests' : [0,1],
                        'attractive_important' : [0,1],
                        'sincere_important' : [0,1],
                        'intelligence_important' : [0,1],
                        'funny_important' : [0,1],
                        'ambition_important' : [0,1],
                        'shared_interests_important' : [0,1],
                        'attractive' : [0,10],
                        'sincere' : [0,10],
                        'intelligence' : [0,10],
                        'funny' : [0,10],
                        'ambition' : [0,10],
                        'attractive_partner':[0,10],
                        'sincere_partner':[0,10],
                        'intelligence_parter':[0,10],
                        'funny_partner':[0,10],
                        'ambition_partner':[0,10],
                        'shared_interests_partner':[0,10],
                        'sports':[0,10],
                        'tvsports': [0,10],
                        'exercise': [0,10],
                        'dining':   [0,10],
                        'museums':  [0,10],
                        'art':      [0,10],
                        'hiking':   [0,10],
                        'gaming': [0,10],
                        'clubbing': [0,10],
                        'reading': [0,10],
                        'tv': [0,10],
                        'theater': [0,10],
                        'concerts': [0,10],
                        'music': [0,10],
                        'shopping': [0,10],
                        'yoga': [0,10],
                        'interests_correlate': [-1,1],
                        'expected_happy_with_sd_people': [0,10],
                        'like': [0,10],
                        'movies': [0,10],
                        }

    for column in data.columns:
        if column not in excludedColumns:
            #convert any values greater than the maximum value of the attribute to the maximum value
            data[column] = data[column].apply(lambda x : rangeDictionary[column][1] if x > rangeDictionary[column][1] else x)
            min_value = rangeDictionary[column][0]
            max_value = rangeDictionary[column][1]
            bin_range = (max_value - min_value)/bins
            binValues = [min_value + (i*bin_range) for i in range(0,bins+1)]
            data[column]=pd.cut(data[column],binValues,include_lowest=True,labels=list(range(0,bins)))
    
    return data

#Dropping the columns
data = data.drop(columns=['race', 'race_o' ,'field'])

#1(iii)
data['gender'],encodedMapping = data['gender'].factorize(sort=True)

#1(iv)
preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive','pref_o_sincere', 'pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

temp_data = data[preference_scores_of_participant].copy()

temp_data['total'] = 0

for column in preference_scores_of_participant:
    temp_data['total'] += data[column] 

for column in preference_scores_of_participant:
    data[column] = data[column] / temp_data['total']

temp_data['total'] = 0

for column in preference_scores_of_partner:
    temp_data['total'] += data[column] 

for column in preference_scores_of_partner:
    data[column] = data[column] / temp_data['total']

#discretize to two bins
data = discretize(data,2)

test = data.sample(random_state=47,frac=0.2)
train = data.drop(test.index)

train.to_csv('trainingSet.csv',index=False)
test.to_csv('testSet.csv',index=False)