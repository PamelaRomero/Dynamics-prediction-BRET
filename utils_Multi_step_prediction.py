import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from metadata import Dose, LigandOHE, ReceptorOHE, PerturbationOHE
import time



def Dataset(path):
    time_list, signal_list, names = Main(path)
    dose_list, name_list = Dose(names)
    LigandOHE_list = LigandOHE(names)
    ReceptorOHE_list = ReceptorOHE(names)
    PerturbationOHE_list = PerturbationOHE(names)
    data, metadata = Data_Metadata(time_list, signal_list, dose_list, LigandOHE_list, ReceptorOHE_list, PerturbationOHE_list)

    print('dose_list: ', len(dose_list))
    print('LigandOHE_list: ', len(LigandOHE_list))
    print('ReceptorOHE_list: ', len(ReceptorOHE_list))
    print('PerturbationOHE_list: ', len(PerturbationOHE_list))

    return data, metadata, name_list


def Main(path):
    list_path = paths(path)
    names, signal, signal_list, time, time_list = list(), list(), list(), list(), list()
    count = 0
   
    for i in range(len(list_path)):
        file_on = list_path[i]
        file_read = pd.ExcelFile(file_on)
        sheet_names = file_read.sheet_names
        df = pd.read_excel(file_on, sheet_names[0])

        rows, columns = df.shape
        row_Time, col_Time = InitialRowColumn(df)
        for j in range(col_Time+1, columns):
            for k in range(row_Time, rows): 
                if (k > row_Time):
                    signal.append(float(df.iloc[k, j]))  
                    time.append(float(df.iloc[k, col_Time]))
                else:
                    if count == 0:
                        names.append(df.iloc[k, j])
                        count = count + 1
                    elif count == 1:
                        names.append(names[-1])
                        count = count + 1
                    else:
                        names.append(names[-1])
                        count = 0
            signal_list.append(signal)
            time_list.append(time)
            signal = []
            time = []
   
    return time_list, signal_list, names


def InitialRowColumn(df):
    rows, columns = df.shape
    for k in range(0, rows): 
        for l in range(0, columns):
            if(df.iloc[k,l] == 'Induced BRET') or (df.iloc[k,l] == 'Induced BRET normalized on baseline'):
                k2, l2 = k, l
            else:
                pass

    for k1 in range(k2, rows): 
        for l1 in range(l2, columns):
            if (df.iloc[k1,l1] == 'Time (min)')  or (df.iloc[k1,l1] == 'Time WASH (min)') or (df.iloc[k1,l1]) == 'Time NO WASH (min)' or (df.iloc[k1,l1] == 'Time (min) for WASH') or (df.iloc[k1,l1] == 'Time NO WASH (min)'):  
                row_Time, col_Time = k1, l1
                break                
            else:
                pass
    return row_Time, col_Time


def Data_Metadata(time_list, signal_list, dose_list, ligand_list, receptor_list, perturbation_list):
    metadata = list()
    data = list()
    for i in range(len(dose_list)):
        metadata.append(dict(dose=dose_list[i], ligand=ligand_list[i], receptor=receptor_list[i], perturbation=perturbation_list[i]))
    metadata = pd.DataFrame(metadata)
    for j in range(len(signal_list)):
        data.append(np.array([time_list[j], signal_list[j]]))
    return data, metadata


def Nivelar_min(data):
    lenn = list()
    for i in range(len(data)):
        length = len(data[i][0])
        lenn.append(length)
    min_lista = min(lenn)
    new_data = list()
    for j in range(len(data)):
        length2 = len(data[j][0])
        if length2 != min_lista:
            new_data.append([data[j][0][:min_lista], data[j][1][:min_lista]])
        else:
            new_data.append([data[j][0][:min_lista], data[j][1][:min_lista]])
    return new_data


def SplitTrainValidTest(data, metadata, name_lista, trn_pc, val_pc, tst_pc):
    new_data = int(len(data) / 3)
    to_trn = round(int(new_data * trn_pc) * 3)
    to_val = round(int(new_data * (trn_pc + val_pc)) * 3)

    trn_data = data[:to_trn]
    val_data = data[to_trn:to_val]
    tst_data = data[to_val:]

    print('trn_data: ',len(trn_data))
    print('val_data: ', len(val_data))
    print('tst_data: ', len(tst_data))

    trn_name = CadaTres(name_lista[:to_trn])
    val_name = CadaTres(name_lista[to_trn:to_val])
    tst_name = CadaTres(name_lista[to_val:])

    trn_metadata = metadata[:to_trn]
    val_metadata = metadata[to_trn:to_val]
    tst_metadata = metadata[to_val:]
    
    listData_trn, metaData_trn = Dataa(trn_data, trn_metadata)
    listData_val, metaData_val = Dataa(val_data, val_metadata)
    listData_tst, metaData_tst = Dataa(tst_data, tst_metadata)
    
    return listData_trn, listData_val, listData_tst, trn_name, val_name, tst_name, metaData_trn, metaData_val, metaData_tst
        

def CadaTres(name):
    name_final = list()
    for i in range(0,len(name),3):
        name_final.append(name[i])
    return name_final


def Dataa(data, metadata):
    listaData = list()
    listaa = list()
    for i in range(0, len(data)-2, 3):
        listaa.append(i)
        Bret = SumaListas(data[i][1], data[i+1][1], data[i+2][1])
        Time = SumaListas(data[i][0], data[i+1][0], data[i+2][0])
        listaData.append([np.array(Time), np.array(Bret)])
    return listaData, metadata.iloc[listaa]
    

def SumaListas(lista1, lista2, lista3):
    lista4 = list()
    for i in range(len(lista1)):
        suma = (lista1[i] + lista2[i] + lista3[i]) / 3
        lista4.append(round(suma,3))
    return lista4


def Data(dataa, metadatos, c):
    res = metadatos['receptor'].to_list()
    lig = metadatos['ligand'].to_list()
    dos = metadatos['dose'].to_list()
    per = metadatos['perturbation'].to_list()

    BRET_Y, BRET, TIME, DOSE, LIGAND, RECEPTOR, PERTURBATION, cantidad, TIME_Y = list(), list(), list(), list(), list(), list(), list(), list(), list()  
    
    for i in range(len(dataa)):
        t, x = dataa[i][0], dataa[i][1]

        cant = int(round(c * len(x)))
        BRET_Y.append(x[cant:].tolist())

        DOSE.append(dos[i])
        LIGAND.append(lig[i])
        RECEPTOR.append(res[i])
        PERTURBATION.append(per[i])

        cantidad.append(cant)
    
        BRET.append(x[:cant].tolist())
        TIME.append(t[:cant].tolist())
        TIME_Y.append(t[cant:].tolist())       
       
    return TIME, BRET, BRET_Y, DOSE, LIGAND, RECEPTOR, PERTURBATION, cantidad, TIME_Y


def ToList(x):
    return list(itertools.chain.from_iterable(x))


def Vector(time, x_data, dose, ligand, receptor, perturbation, Dos, Lig, Rec, Per):
    vector, vectores, x_list, time_list, meta = list(), list(), list(), list(), list()

    if (Dos == 1) and (Lig == 1) and (Rec == 1) and (Per == 1):

        for k in range(len(x_data)):
            meta = [dose[k], ligand[k][0],ligand[k][1],ligand[k][2],ligand[k][3],receptor[k][0],receptor[k][1],receptor[k][2],receptor[k][3],receptor[k][4],receptor[k][5], perturbation[k][0],perturbation[k][1],perturbation[k][2],perturbation[k][3],perturbation[k][4],perturbation[k][5],perturbation[k][6],perturbation[k][7],perturbation[k][8]]
            
            for xd in range(len(x_data[k])):
                x_list.append(x_data[k][xd])
                time_list.append(time[k][xd])

            vector = list(itertools.chain(meta, time_list, x_list))
            x_list = []
            time_list = []
            meta = []

            vectores.append(vector)
            vector = []       
            
    elif (Dos == 1) and (Lig == 1) and (Rec == 0) and (Per == 0):
        for k in range(len(x_data)):
            meta = [dose[k], ligand[k][0],ligand[k][1],ligand[k][2],ligand[k][3],ligand[k][4],ligand[k][5],ligand[k][6],ligand[k][7],ligand[k][8],ligand[k][9]]
            
            for xd in range(len(x_data[k])):
                x_list.append(x_data[k][xd])
                time_list.append(time[k][xd])
            
            vector = list(itertools.chain(meta, time_list, x_list))
            x_list = []
            time_list = []
            meta = []

            vectores.append(vector)
            vector = []
    else:
        pass
    return vectores


def paths(path):    
    files = os.listdir(path)  
    list_path = list()
    for file in files:         
        files2 = os.listdir(path + '/' + file)
        for file2 in files2:
            if '.xlsx' not in file2:
                files3 = os.listdir(path + '/' + file + '/' + file2)
                for file3 in files3:
                    list_path.append(path + '/' + file + '/' + file2 + '/' + file3)
            else:
                list_path.append(path + '/' + file + '/' + file2)
    return list_path


def NormalizeMinMax2(x_data_train):
    ma_lista = list()
    mi_lista = list()
    x_new = list()
    for i in range(len(x_data_train)):
        ma_lista.append(np.max(x_data_train[i]))  
        mi_lista.append(np.min(x_data_train[i]))

    maa = np.max(ma_lista)
    mii = np.min(mi_lista)

    for j in range(len(x_data_train)):
        x_new.append((x_data_train[j] - mii) / (maa - mii))
    return x_new, maa, mii


def NormalizeMinMaxTest2(x, maax, miin):
    x_new = list()
    for j in range(len(x)):
        x_new.append((x[j] - miin) / (maax - miin))
    return x_new

 
def FormaY(vector):
        l1, l2 = list(), list()
        for i in range(len(vector)):
            for j in range(len(vector[i])):
                l1.append((vector[i][j].tolist()))
            l2.append(l1)
            l1 = []
        return l2  


def FormaV(vector):
        l1, l2 = list(), list()
        for i in range(len(vector)):
            for j in range(len(vector[i])):
                l1.append((vector[i][j]))
            l2.append(l1)
            l1 = []
        return l2  


def Acc2(l1, l2): 
    mape_sum, mape = 0, 0
    for i in range(len(l1)):
        mape = MAPEE(l1[i], l2[i])
        mape_sum = mape_sum + mape
        mape = 0
    mape_ofi = round(mape_sum / len(l1),4)
    print("mape: ", mape_ofi)
    print('acc: ', 100.0 - mape_ofi)   


def MAE(gt, pred):
    lista = list()
    for i in range(len(gt)):
        lista.append(np.mean(np.abs([e1 - e2 for e1, e2 in zip(gt[i],pred[i])])))
    print('Mean Absolute Error (MAE):', round(sum(lista)/len(lista),4))


def MSE(gt, pred):
    lista = list()
    for i in range(len(gt)):
        lista.append(np.mean([e1 - e2 for e1, e2 in zip(gt[i],pred[i])])**2) 
    print('Mean Squared Error (MSE):', round(sum(lista)/len(lista),4))


def MAPEE(y_true, y_pred):
    #REFERENCE: https://stackoverflow.com/questions/47648133/mape-calculation-in-python
    actual = np.asarray(y_true) 
    predicted = np.asarray(y_pred)
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return np.mean(np.abs(res)) * 100


def Metrics(gt, pred):
    MAE(gt, pred)
    MSE(gt, pred)
    Acc2(gt, pred)  
  

def FIMDI_1(forest):
    #REFERENCE: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    feature_names = ['Dose','L1','L2','L3','L4','R1','R2','R3','R4','R5','R6','P1','P2','P3','P4','P5','P6','P7','P8','P9','T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','X0','X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15']
    
    forest_importances = pd.Series(importances, index=feature_names)
   
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.grid()
    plt.savefig("/home/.../Results/Multi_time_step_model/FIMDI.png", dpi=200)






