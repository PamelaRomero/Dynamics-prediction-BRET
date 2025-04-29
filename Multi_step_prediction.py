import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from utils_Multi_step_prediction import SplitTrainValidTest, Dataset, Data, Vector, NormalizeMinMaxTest2, NormalizeMinMax2, Metrics, Nivelar_min, FormaY, FormaV, FIMDI_1
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree  



def Normalization(x_data_trn, y_data_trn):
    x_data_trn1, x_data_trn_max, x_data_trn_min = NormalizeMinMax2(x_data_trn)
    y_data_trn1, y_data_trn_max, y_data_trn_min = NormalizeMinMax2(y_data_trn)

    lis = [x_data_trn_max, x_data_trn_min, y_data_trn_max, y_data_trn_min]
    z_min = min(lis)
    z_max = max(lis)

    return z_min, z_max

 
def Principal(data_trn, metadata_trn, data_val, metadata_val, data_tst, metadata_tst, name_list_val, name_list_tst, name_list_trn, D, L, R, P, cant, folder, start):
    Data_type = object

    time_trn, x_trn, y_trn, D_trn, L_trn, R_trn, P_trn, cant_trn, time_y_trn = Data(data_trn, metadata_trn, cant)         
    zmin, zmax = Normalization(x_trn, y_trn)
    x_data_trn = NormalizeMinMaxTest2(x_trn, zmax, zmin)
    y_trn = np.array(FormaY(NormalizeMinMaxTest2(y_trn, zmax, zmin)), dtype=Data_type)
    D_trn, dose_trn_max, dose_trn_min = NormalizeMinMax2(D_trn)
    vec_trn = np.array(FormaV(Vector(time_trn, x_data_trn, D_trn, L_trn, R_trn, P_trn, D, L, R, P)), dtype=Data_type) 
    
    time_val, x_val, y_val, dose_val, L_val, R_val, P_val, cant_val, time_y_val = Data(data_val, metadata_val, cant)
    x_data_val = NormalizeMinMaxTest2(x_val, zmax, zmin)
    y_val = np.array(FormaY(NormalizeMinMaxTest2(y_val, zmax, zmin)), dtype=Data_type)
    dose_val = NormalizeMinMaxTest2(dose_val, dose_trn_max, dose_trn_min)
    vec_val = np.array(FormaV(Vector(time_val, x_data_val, dose_val, L_val, R_val, P_val, D, L, R, P)), dtype=Data_type) 
    
    time_tst, x_tst, y_tst, D_tst, L_tst, R_tst, P_tst, cant_tst, time_y_tst= Data(data_tst, metadata_tst, cant)
    x_data_tst = NormalizeMinMaxTest2(x_tst, zmax, zmin )
    y_tst = np.array(FormaY(NormalizeMinMaxTest2(y_tst, zmax, zmin)), dtype=Data_type)
    D_tst = NormalizeMinMaxTest2(D_tst, dose_trn_max, dose_trn_min)
    vec_tst = np.array(FormaV(Vector(time_tst, x_data_tst, D_tst, L_tst, R_tst, P_tst, D, L, R, P)), dtype=Data_type)

    #GridSearch(vec_trn, y_trn)
    
    regressor = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1, random_state=0)
    #regressor = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=1, random_state=0)
    regressor.fit(vec_trn, y_trn)
    
    FIMDI_1(regressor)
    
    print("-------------Train-RF------------")
    Ypred_trn = regressor.predict(vec_trn)
    Metrics(y_trn, Ypred_trn)

    print("-------------VAL-RF------------")
    Ypred_val = regressor.predict(vec_val) 
    Metrics(y_val, Ypred_val)

    print("-------------Test-RF------------")
    Ypred_tst = regressor.predict(vec_tst) 
    Metrics(y_tst, Ypred_tst)

    end = time.time()
    print('Time: ', end-start)
  
    porc =50
    
    np.save(folder +'y_tst_{}.npy'.format(porc), y_tst)    
    np.save(folder +'time_tst_{}.npy'.format(porc), time_tst)  
    np.save(folder +'time_y_tst_{}.npy'.format(porc), time_y_tst)  
    np.save(folder +'x_data_tst_{}.npy'.format(porc), x_data_tst)  
    np.save(folder +'Ypred_tst_{}.npy'.format(porc), Ypred_tst)  
    np.save(folder +'name_list_tst_{}.npy'.format(porc), name_list_tst)  
    
    Plott(y_val, time_val, time_y_val, x_data_val, Ypred_val, name_list_val, folder +"Val")
    Plott(y_tst, time_tst, time_y_tst, x_data_tst, Ypred_tst, name_list_tst, folder +"Test")
    Plott(y_trn, time_trn, time_y_trn, x_data_trn, Ypred_trn, name_list_trn, folder + "Train")
   

def GridSearch(vec_trn, y_trn):
    params_grid = {
        'max_depth': [1, 3, 5, 10, 15],
        'min_samples_leaf': [1, 3, 5, 10, 15],
        #'min_samples_split': [1, 3, 5, 10, 15],
        'n_estimators': [1, 5, 10] 
    }
    #regre = RandomForestRegressor(random_state=0)
    regre = DecisionTreeRegressor(random_state=0)
    regressor = GridSearchCV(estimator=regre, param_grid=params_grid, cv=5, scoring='neg_mean_squared_error')
    regressor.fit(vec_trn, y_trn)
    print("-------------- GridSearchCV --------------")
    print(regressor.best_params_)
    print("------------------------------------------")


def Plott(y, time, time_y, x, Ypred, name_list, folder_name):
    for p in range(len(y)):
        plt.figure()
        plt.plot(time[p], x[p], color = 'red', label='Ground Truth')
        plt.plot([time[p][-1], time_y[p][0]], [x[p][-1], y[p][0]], color = 'red')
        plt.plot(time_y[p], y[p], color = 'red')
        plt.plot(time_y[p], Ypred[p], color = 'blue', label='Random Forest Regressor')  

        plt.legend()
        plt.title('{}'.format(name_list[p]))
        plt.ylabel('BRET Ratio')
        plt.xlabel('Time')
        plt.savefig("{}/fig_{}.png".format(folder_name, p), dpi=200)
        plt.close()



start = time.time()
path = '/home/.../Datanew'
data, metadata, names = Dataset(path) 

new_data = Nivelar_min(data)
train_pc, valid_pc, test_pc = 0.7, 0.1, 0.2
listData_trn, listData_val, listData_tst, trn_name, val_name, tst_name, metaData_trn, metaData_val, metaData_tst = SplitTrainValidTest(new_data, metadata, names, train_pc, valid_pc, test_pc)

print('listData_trn: ', len(listData_trn))
print('listData_val: ', len(listData_val))
print('listData_tst: ', len(listData_tst))

D, L, R, P, cant = 1, 1, 1, 1, 0.5

folder = "/home/.../Multi_time_step_model/{}/".format(int(cant*100))

print("D:",D, "L:",L,", R:",R,", P:",P,", %:",cant)
Principal(listData_trn, metaData_trn, listData_val, metaData_val, listData_tst, metaData_tst, val_name, tst_name, trn_name, D, L, R, P, cant, folder, start)

