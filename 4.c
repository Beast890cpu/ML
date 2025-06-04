import pandas as pd 
 
data = pd.read_csv(r"C:\Users\vijay\Desktop\Machine Learning Course 
Batches\FDP_ML_6thSem_VTU\Experiment_4_FindS\training_data.csv") 
 
print(data)

def find_s_algorithm(data): 

 for i in range(len(target)): 
        if target[i] == "Yes":  
            hypothesis = attributes[i].copy() 
            break

  for i in range(len(target)): 
        if target[i] == "Yes": 
            for j in range(len(hypothesis)): 
                if hypothesis[j] != attributes[i][j]: 
                    hypothesis[j] = '?'  
     
    return hypothesis  
 
final_hypothesis = find_s_algorithm(data) 
 

print("Most Specific Hypothesis:", final_hypothesis) 
